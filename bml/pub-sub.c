#define _XOPEN_SOURCE   600
#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include "pub-sub.h"
#include "utils.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <limits.h>

// -------------------------------
// Handle Stop Signal
// -------------------------------
#include <signal.h>
#include <stdatomic.h>

volatile sig_atomic_t dump_stats = 0;

void sigusr1_handler(int signum) {
    if (signum == SIGUSR1) dump_stats = 1;
}

// -------------------------------
// Configuration
// -------------------------------
#define MAX_RETRIES 1000
#define RETRIES_INTERVAL 1000
#define MAX_SUBSCRIBERS 100

// Inputs with default values
char* library_name = NULL;
int is_logging_enabled = 0;
FILE* log_file = NULL;
int use_config_file = 0;
char* config_filename = "config.yaml";
int message_count = INT_MAX;
int payload_length = 100;
char endpoint[256] = "inproc://benchmark";
int pub_time_interval = 0;
int pub_start_delay = 0;
subscriber_args_t subscribers[MAX_SUBSCRIBERS];

// Auxiliary
char* payload;
int subscriber_ready = 0;
int publisher_ready = 0;

// Handle pub and subs logic
int run_as_publisher = -1; // -1: not set (in-proc), 0: subscriber, 1: publisher
static int sub_count = 0;
static int pub_count = 0;

void display_help(char* program) {
    printf("Usage: %s <zeromq|nanomsg|nng> [OPTIONS]\n"
           "Examples:\n"
           "  In-process:\t\t%s zeromq --log --sub --sub --sub_ids sub1 sub2 --pub --count 100 --rate 500 --dp-len 100 --delay 1000 --endpoint inproc://benchmark\n"
           "  Inter-process:\n"
           "    Publisher:\t%s zeromq --log --pub --count 100 --rate 500 --dp-len 100 --delay 1000 --endpoint ipc://127.0.0.1:5555\n"
           "    Subscriber:\t%s zeromq --log --sub --sub_ids sub1 --count 100 --endpoint ipc://127.0.0.1:5555\n"
           "\n"
           "  --help                         Show this menu\n"
           "  -z, --conf <file>              Use config file for options (config has priority)\n"
           "  -l, --log                      Enable logging (debugging)\n"
           "  -c, --count <number>           Total number of messages to be published (default: %d)\n"
           "  -f, --rate <microseconds>      Publishing time interval in microseconds (default: %d)\n"
           "  -d, --dp-len <length>          Length of the datapoint (payload) (default: %d)\n"
           "  -y, --delay <milliseconds>     Initial delay before publisher starts (default: %d)\n"
           "  -e, --endpoint <endpoint>      Endpoint for publisher and subscriber (default: %s)\n"
           "  --pub                          Run as publisher (inter-process communication only)\n"
           "  --sub                          Run as subscriber (inter-process communication only)\n"
           "  --sub_ids <id1> <id2> ...      Provide IDs for each subscriber if multiple subs are used\n"
           , program, program, program, program, message_count, pub_time_interval, payload_length, pub_start_delay, endpoint);
    exit(EXIT_FAILURE);
}

void open_log_file(int is_publisher) {
    uint64_t timestamp = get_timestamp();
    
    // create logs directory if it doesn't exist
    struct stat st = {0};
    if (stat("logs", &st) == -1) {
        mkdir("logs", 0700);
    }

    char log_filename[256];
    if (is_publisher == -1) 
        snprintf(log_filename, 255, "logs/log_%d_%d_%lu.log", message_count, payload_length, timestamp);
    else if (is_publisher == 1)
        snprintf(log_filename, 255, "logs/log_pub_%d_%d_%lu.log", message_count, payload_length, timestamp);
    else 
        snprintf(log_filename, 255, "logs/log_sub_%d_%d_%lu.log", message_count, payload_length, timestamp);
    
    log_file = fopen(log_filename, "w");
    if (!log_file) {
        fprintf(stderr, "Failed to open log file\n");
        exit(EXIT_FAILURE);
    }
}

void close_log_file() {
    if (log_file) {
        fflush(log_file);
        fclose(log_file);
    }
}

void read_config(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "COUNT: %d", &message_count) == 1) continue;
        if (sscanf(line, "DP-LEN: %d", &payload_length) == 1) continue;
        if (sscanf(line, "RATE: %d", &pub_time_interval) == 1) continue;
        if (sscanf(line, "DELAY: %d", &pub_start_delay) == 1) continue;
        if (sscanf(line, "ENDPOINT: %s", endpoint) == 1) continue;
    }

    fclose(file);
}


void generate_payload() {
    if (payload_length < TIMESTAMP_SIZE) {
        payload = (char*)malloc(1);
        if (!payload) {
            fprintf(stderr, "Failed to allocate memory for payload\n");
            exit(EXIT_FAILURE);
        }
        payload[0] = '\0';
    } else {
        payload = (char*)malloc(payload_length - TIMESTAMP_SIZE + 1);
        if (!payload) {
            fprintf(stderr, "Failed to allocate memory for payload\n");
            exit(EXIT_FAILURE);
        }
        memset(payload, 'A', payload_length - TIMESTAMP_SIZE);
        payload[payload_length - TIMESTAMP_SIZE] = '\0';
    }
}

char* get_message() {
    size_t message_size = strlen(payload) + TIMESTAMP_SIZE + 1;
    char* message = (char*)malloc(message_size);
    if (!message) {
        fprintf(stderr, "Failed to allocate memory for message\n");
        return NULL;
    }
    snprintf(message, message_size, "%lu|%s", get_timestamp(), payload);
    return message;
}

uint64_t parse_message(char* message) {
    char* delim_ptr = strrchr(message, '|');
    if (delim_ptr) {
        *delim_ptr = '\0';
        return strtoull(message, NULL, 10);
    } else {
        fprintf(stderr, "Failed to parse message: delimiter not found\n");
        return 0;
    }
}

// -------------------------------
// Publisher and Subscriber
// -------------------------------

void publisher(void *pipe, void *args) {
    prctl(PR_SET_NAME, "bml_publisher", 0, 0, 0);
    (void)args;  // unused parameter
    int is_publisher = 1;
    void* pub_poller = NULL;
    int retries = 0;

    if (is_logging_enabled)
        fprintf(log_file, "Starting publisher\n");

    void* pub = messaging_ops.create_pub(endpoint, pipe);
    if (!pub) {
        fprintf(stderr, "Failed to create publisher\n");
        return;
    }

    if (polling_enabled) {
        pub_poller = messaging_ops.create_poller(pub, is_publisher);
        if (!pub_poller) {
            fprintf(stderr, "Failed to create publisher poller\n");
            messaging_ops.destroy(pub);
            return;
        }
    }
    
    if (pub_start_delay > 0) usleep(pub_start_delay * 1000);

    for (int count = 1; count <= message_count; count++) {
        char* message = get_message();
        if (!message) break;
        if (is_logging_enabled) fprintf(log_file, "Sending message: %s\n", message);

        if (polling_enabled) {
           int poll_result = messaging_ops.poll_socket(pub_poller, is_publisher);
            if (poll_result == 0) {
                retries = 0;
                messaging_ops.send(pub, message);
            } else {
                retries++;
                if (retries > MAX_RETRIES || dump_stats) {
                    fprintf(stderr, "Pub: Exiting from polling loop.\n");
                    messaging_ops.free(message);
                    break;
                }
                fprintf(stderr, "Retrying message %d\n", count);
                usleep(RETRIES_INTERVAL);
                count--;
                messaging_ops.free(message);
                continue;
            }
        }
        else {
            messaging_ops.send(pub, message);
        }
        messaging_ops.free(message);
        usleep(pub_time_interval);
        if (dump_stats) break;
    }
    messaging_ops.destroy(pub);
    free(pub_poller);
}

void subscriber(void *pipe, void *args) {
    subscriber_args_t* sargs = (subscriber_args_t*)args;
    char* sub_id = sargs->id;

    char prctl_name[256];
    snprintf(prctl_name, sizeof(prctl_name), "bml_subscriber_%s", sub_id);
    prctl(PR_SET_NAME, prctl_name, 0, 0, 0);

    int is_publisher = 0;
    void* sub_poller = NULL;
    int target_message_count = message_count;

    if (is_logging_enabled)
        fprintf(log_file, "Starting subscriber %s\n", sub_id);
    
    void* sub = messaging_ops.create_sub(endpoint, "", pipe);
    if (!sub) {
        fprintf(stderr, "Failed to create subscriber\n");
        return;
    }

    if (polling_enabled) {
        sub_poller = messaging_ops.create_poller(sub, is_publisher);
        if (!sub_poller) {
            fprintf(stderr, "Failed to create subscriber poller\n");
            messaging_ops.destroy(sub);
            return;
        }
    }
    
    uint64_t* latencies = (uint64_t*)malloc(target_message_count * sizeof(uint64_t));
    if (!latencies) {
        fprintf(stderr, "Failed to allocate memory for latencies\n");
        messaging_ops.destroy(sub);
        return;
    }

    int messages_received = 0;
    int messages_lost = 0;
    int retries = 0;

    uint64_t first_pub_timestamp = 0;
    uint64_t last_sub_timestamp = 0;
    
    // the first polling has a higher timeout to allow the publisher to connect
    polling_timeout += pub_start_delay;

    while ((messages_received + messages_lost) < target_message_count) {
        if (polling_enabled) {
            int poll_result = messaging_ops.poll_socket(sub_poller, is_publisher);
            if (poll_result != 0) {
                retries++;
                if (retries > MAX_RETRIES || dump_stats) {
                    fprintf(stderr, "Sub: Exiting from polling loop.\n");
                    break;
                }
                usleep(RETRIES_INTERVAL);
                continue;
            }
        }

        // infer payload length from the message received
        char* message = messaging_ops.recv(sub);
        if (!message) break;
        else payload_length = strlen(message); 

        int64_t sub_timestamp = get_timestamp();
        int64_t pub_timestamp = parse_message(message);
        
        int64_t latency = sub_timestamp - pub_timestamp;
        latencies[messages_received] = latency;

        if (is_logging_enabled) {
            char short_sub_id[6] = {sub_id[0], sub_id[1], sub_id[2], sub_id[3], sub_id[4], '\0'};
            fprintf(log_file, "(%s) Received timestamp: %lu, current timestamp: %lu, latency: %lu\n", short_sub_id, pub_timestamp, sub_timestamp, latency);
        }

        if (pub_timestamp != -1) {
            if (messages_received == 0) {
                first_pub_timestamp = pub_timestamp;
                // reset the timeout to the original value
                polling_timeout -= pub_start_delay;
            }
            last_sub_timestamp = sub_timestamp;
            messages_received++;
        }
        else {
            messages_lost++;   
        }
        messaging_ops.free(message);
        if (dump_stats) break;
    }
    messaging_ops.destroy(sub);
    compute_metrics(sub_id, latencies, (messages_received + messages_lost), messages_received, 
                    first_pub_timestamp, last_sub_timestamp, 
                    payload_length);
    free(latencies);
    free(sub_poller);
}


// -------------------------------
// Main
// -------------------------------

extern messaging_ops_t ops_zeromq;
extern messaging_ops_t ops_nanomsg;
extern messaging_ops_t ops_nng;
messaging_ops_t messaging_ops;

void process_init(void) {
    printf("Initializing processes\n");
    if (run_as_publisher == -1) {
        fprintf(stderr, "Error: Please specify --pub or --sub when using inter-process communication.\n");
        exit(EXIT_FAILURE);
    }
    if (run_as_publisher == 1) {
        printf("Running as publisher\n");
        if (is_logging_enabled) open_log_file(1);
        publisher(NULL, NULL);
        if (is_logging_enabled) close_log_file();
    } else {
        printf("Running as subscriber\n");
        if (is_logging_enabled) open_log_file(0);
        subscriber(NULL, &subscribers[0]);
        if (is_logging_enabled) close_log_file();
    }
}

void threads_init() {
    printf("Initializing threads\n");
    if (is_logging_enabled) open_log_file(-1);
    messaging_ops.threads_init(sub_count, subscribers);
    if (is_logging_enabled) close_log_file();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        display_help(argv[0]);
    }

    signal(SIGUSR1, sigusr1_handler); 

    if (strcmp(argv[1], "zeromq") == 0) {
        printf("Using ZeroMQ, ");
        messaging_ops = ops_zeromq;
        library_name = "zeromq";
    } else if (strcmp(argv[1], "nanomsg") == 0) {
        printf("Using NanoMsg, ");
        messaging_ops = ops_nanomsg;
        library_name = "nanomsg";
    } else if (strcmp(argv[1], "nng") == 0) {
        printf("Using NNG, ");
        messaging_ops = ops_nng;
        library_name = "nng";
        polling_enabled = 0; // disable polling for NNG
    } else {
        fprintf(stderr, "Unknown library: %s\n", argv[1]);
        display_help(argv[0]);
    }

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            display_help(argv[0]);
        } else if (strcmp(argv[i], "--log") == 0) {
            is_logging_enabled = 1;
        } else if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            use_config_file = 1;
            config_filename = argv[++i];
            read_config(config_filename);
            break;
        } else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            message_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            pub_time_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dp-len") == 0 && i + 1 < argc) {
            payload_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            pub_start_delay = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--endpoint") == 0 && i + 1 < argc) {
            strncpy(endpoint, argv[++i], sizeof(endpoint) - 1);
        } else if (strcmp(argv[i], "--pub") == 0) {
            run_as_publisher = 1;
            pub_count++;
            if (pub_count > 1) {
                fprintf(stderr, "Only one publisher is allowed\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--sub") == 0) {
            run_as_publisher = 0;
            sub_count++;
        } else if (strcmp(argv[i], "--sub_ids") == 0) {
            if (sub_count == 0) {
                fprintf(stderr, "No --sub flags before --sub_ids\n");
                exit(EXIT_FAILURE);
            }
            if (i + sub_count >= argc) {
                fprintf(stderr, "Not enough subscriber IDs provided\n");
                exit(EXIT_FAILURE);
            }
            for (int s = 0; s < sub_count; s++) {
                i++;
                char* sub_id = argv[i];
                strncpy(subscribers[s].id, sub_id, sizeof(subscribers[s].id) - 1);
            }
        } else if (strcmp(argv[i], "--dump_latencies") == 0) {
            latency_dump_enabled = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            display_help(argv[0]);
        }
    }
    generate_payload();

    printf("polling is %s, ", polling_enabled ? "enabled" : "disabled");
    if (strcmp(get_transport(endpoint), "inproc") == 0) {
        printf("and inproc transport\n");
        printf("Benchmark for pub-sub using %s\n", argv[1]);
        if (pub_count == 0) {
            fprintf(stderr, "No publisher specified\n");
            exit(EXIT_FAILURE);
        }
        threads_init();
    }
    else if (strcmp(get_transport(endpoint), "ipc") == 0 || strcmp(get_transport(endpoint), "tcp") == 0) {
        printf("and %s transport\n", get_transport(endpoint));
        printf("Benchmark for pub-sub using %s\n", argv[1]);
        process_init();
    }
    else {
        fprintf(stderr, "Unknown transport for %s\n", endpoint);
        exit(EXIT_FAILURE);
    }

    if (payload_length < TIMESTAMP_SIZE) {
        fprintf(stderr, "\033[1;33mWarning: The payload length will be at least %d bytes to carry the timestamps\033[0m\n", TIMESTAMP_SIZE);
    }

    return EXIT_SUCCESS;
}
