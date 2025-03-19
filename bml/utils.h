#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

#define TIMESTAMP_SIZE 20 // 19 digits + 1 null terminator

uint64_t get_timestamp();

char* get_transport(char *endpoint);

int cmpfunc(const void *a, const void *b);

void compute_metrics(char* id, uint64_t* latencies, 
                     int messages_sent, int message_received, 
                     uint64_t first_pub_timestamp, uint64_t last_sub_timestamp, 
                     int payload_length);

void dump_latencies_to_binary(char* id, uint64_t* latencies, int count);

void log_results_to_json(char* id, int message_received, double total_time_sec,
                         double throughput, double payload_throughput,
                         uint64_t min_latency, uint64_t avg_latency, uint64_t p90_latency, uint64_t p99_latency, uint64_t max_latency, 
                         uint64_t mean_jitter);

extern int polling_enabled;
extern int polling_timeout;
extern int latency_dump_enabled;

#endif // UTILS_H
