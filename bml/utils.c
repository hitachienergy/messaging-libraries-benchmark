#define _POSIX_C_SOURCE 199309L
#include "utils.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>

// Configuration
int polling_enabled = 1; // poll or not
int polling_timeout = 10; // milliseconds
int latency_dump_enabled = 0; // dump latencies to a binary file

/**
 * @brief Get the current timestamp in nanoseconds.
 * 
 * @return uint64_t Current timestamp in nanoseconds.
 */
uint64_t get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}


/**
 * @brief Get the transport type from the endpoint.
 * 
 * @param endpoint Endpoint string.
 * @return char* Transport type.
 */
char* get_transport(char *endpoint) {
    if (strncmp(endpoint, "inproc", 6) == 0) {
        return "inproc";
    }
    else if (strncmp(endpoint, "ipc", 3) == 0) {
        return "ipc";
    }
    else if (strncmp(endpoint, "tcp", 3) == 0) {
        return "tcp";
    }
    else {
        return "unknown";
    }
}


/**
 * @brief Comparison function for sorting uint64_t values.
 * 
 * @param a Pointer to the first value.
 * @param b Pointer to the second value.
 * @return int Negative if a < b, zero if a == b, positive if a > b.
 */
int cmpfunc(const void *a, const void *b) {
    uint64_t val_a = *(const uint64_t*)a;
    uint64_t val_b = *(const uint64_t*)b;
    return (val_a > val_b) - (val_a < val_b);
}


/**
 * @brief Computes and displays performance metrics.
 *
 * Calculates total time, throughput, and latency statistics based on the provided latencies.
 * It computes minimum, average, 90th percentile, 99th percentile, and maximum latencies,
 * as well as jitter and throughput for both publisher and subscriber.
 *
 * @param id                   Identifier of the subscriber.
 * @param latencies            Array of message latencies in nanoseconds.
 * @param messages_sent         Total number of messages expected.
 * @param messages_received     Number of messages received.
 * @param first_pub_timestamp  Timestamp of the first published message.
 * @param last_sub_timestamp   Timestamp of the last received message.
 * @param payload_length       Length of the message payload in bytes.
 */
void compute_metrics(char* id, uint64_t* latencies, 
                     int messages_sent, int messages_received, 
                     uint64_t first_pub_timestamp, uint64_t last_sub_timestamp,
                     int payload_length) { 
    
    if (messages_received == 0) {
        printf("No messages were received.\n");
        return;
    }
    printf("Subscriber %s received %d messages\n", id, messages_received);

    // dump latencies
    if (latency_dump_enabled && strlen(id) > 0)
        dump_latencies_to_binary(id, latencies, messages_received);

    // compute latencies of all messages received, and calculate statistics
    uint64_t total_latency = 0;
    uint64_t max_latency = 0;
    uint64_t min_latency = UINT64_MAX;
    for (int i = 0; i < messages_received; i++) {
        uint64_t latency = latencies[i];
        total_latency += latency;
        if (latency > max_latency) max_latency = latency;
        if (latency < min_latency) min_latency = latency;
    }
    uint64_t avg_latency = total_latency / messages_received;

    // ..including jitter, computed as: 
    // the variation of latencies between consecutive messages (RFC3393)
    uint64_t total_delay_variation = 0;
    for (int i = 1; i < messages_received; i++) {
        uint64_t delay_variation = (latencies[i] > latencies[i - 1]) ? (latencies[i] - latencies[i - 1]) : (latencies[i - 1] - latencies[i]);
        total_delay_variation += delay_variation;
    }
    uint64_t mean_jitter = (messages_received > 1) ? (total_delay_variation / (messages_received - 1)) : 0;

    // ..and percentiles
    qsort(latencies, messages_received, sizeof(uint64_t), cmpfunc);
    uint64_t p90_latency = latencies[(int)(messages_received * 0.9)];
    uint64_t p99_latency = latencies[(int)(messages_received * 0.99)];

    // compute throughput of publisher and subscriber
    double total_time_sec = (last_sub_timestamp - first_pub_timestamp) / 1e9; // seconds

    double throughput = messages_sent / total_time_sec;
    double payload_throughput = throughput * payload_length / 1e6; // MB/s

    printf("\n\n--- Performance Metrics ---\n\n");
    printf("Received ( %d / %d ) messages\n", messages_received, messages_sent);
    printf("Total time: %.9f seconds\n", total_time_sec);
    printf("\nThroughputs:\n\tMessages: %.2f messages/s\n\tPayload: %.2f MB/s\n", throughput, payload_throughput);
    printf("\nLatencies:\n\tMin: %lu ns\n\tAvg: %lu ns\n\tP90: %lu ns\n\tP99: %lu ns\n\tMax: %lu ns\n", min_latency, avg_latency, p90_latency, p99_latency, max_latency);
    printf("\nJitter: %lu ns\n", mean_jitter);

    // dump output metric to a JSON file
    if (strlen(id) > 0)
        log_results_to_json(id, messages_received, total_time_sec, 
                        throughput, payload_throughput, 
                        min_latency, avg_latency, p90_latency, p99_latency, max_latency,
                        mean_jitter);        
}


/**
 * @brief Dumps the latencies to a binary file.
 * 
 * @param id         Identifier of the subscriber.
 * @param latencies  Array of message latencies in nanoseconds.
 * @param count      Number of latencies.
 */
void dump_latencies_to_binary(char* id, uint64_t* latencies, int count) {
    struct stat st = {0};
    if (stat("data", &st) == -1) {
        mkdir("data", 0700);
    }

    char bin_filename[256];
    snprintf(bin_filename, sizeof(bin_filename), "data/%s.bin", id);

    FILE* bin_file = fopen(bin_filename, "wb");
    if (!bin_file) {
        fprintf(stderr, "Failed to open binary latency file\n");
        return;
    }

    fwrite(latencies, sizeof(uint64_t), count, bin_file);
    fclose(bin_file);
}


/**
 * @brief Logs the performance metrics to a JSON file.
 * 
 * @param id                      id of the subscriber
 * @param message_received        Number of messages received.
 * @param total_time_sec          Total time in seconds.
 * @param throughput              Throughput in messages per second.
 * @param payload_throughput      Throughput in MB/s.
 * @param min_latency             Minimum latency in nanoseconds.
 * @param avg_latency             Average latency in nanoseconds.
 * @param p90_latency             90th percentile latency in nanoseconds.
 * @param p99_latency             99th percentile latency in nanoseconds.
 * @param max_latency             Maximum latency in nanoseconds.
 * @param mean_jitter             Mean jitter in nanoseconds.
 */
void log_results_to_json(char* id, int message_received, double total_time_sec,
                         double throughput, double payload_throughput,
                         uint64_t min_latency, uint64_t avg_latency, uint64_t p90_latency, uint64_t p99_latency, uint64_t max_latency, 
                         uint64_t mean_jitter) {
    
    struct stat st = {0};
    if (stat("data", &st) == -1) {
        mkdir("data", 0700);
    }

    char json_filename[256];
    snprintf(json_filename, sizeof(json_filename), "data/%s.json", id);

    FILE* json_file = fopen(json_filename, "w");
    if (!json_file) {
        fprintf(stderr, "Failed to open JSON result file\n");
        return;
    }

    // check for infinite throughput, and print error message
    if (isinf(throughput) || isinf(payload_throughput)) {
        fprintf(json_file, "{\n");
        fprintf(json_file, "  \"message_received\": %d,\n", message_received);
        fprintf(json_file, "  \"time\": %.9f,\n", total_time_sec);
        fprintf(json_file, "  \"error\": \"Infinite throughput\"\n");
        fprintf(json_file, "}\n");
        fclose(json_file);
        return;
    }
    else {
        fprintf(json_file, "{\n");
        fprintf(json_file, "  \"message_received\": %d,\n", message_received);
        fprintf(json_file, "  \"time\": %.9f,\n", total_time_sec);
        fprintf(json_file, "  \"throughput\": %.9f,\n", throughput);
        fprintf(json_file, "  \"payload_throughput\": %.9f,\n", payload_throughput);
        fprintf(json_file, "  \"min_latency\": %lu,\n", min_latency);
        fprintf(json_file, "  \"avg_latency\": %lu,\n", avg_latency);
        fprintf(json_file, "  \"p90_latency\": %lu,\n", p90_latency);
        fprintf(json_file, "  \"p99_latency\": %lu,\n", p99_latency);
        fprintf(json_file, "  \"max_latency\": %lu,\n", max_latency);
        fprintf(json_file, "  \"mean_jitter\": %lu\n", mean_jitter);
        fprintf(json_file, "}\n");
    }
    fclose(json_file);
}