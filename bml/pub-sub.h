#ifndef PUB_SUB_H
#define PUB_SUB_H
// todo: are this lib necessary here?
#include <stdint.h>
#include <czmq.h>

extern int message_count;
extern int payload_length;
extern char* payload;
extern char endpoint[256];
extern int is_logging_enabled;
extern int use_config_file;
extern char* config_filename;
extern int pub_time_interval;
extern int pub_start_delay;

extern int subscriber_ready;
extern int publisher_ready;

// -------------------------------
// Types and Functions
// -------------------------------

typedef struct {
    char id[64];
} subscriber_args_t;

extern subscriber_args_t subscribers[];

typedef struct {
    void (*threads_init)(int num_subs, subscriber_args_t subscribers[]);
    void* (*create_pub)(const char* endpoint, void* pipe);    
    void* (*create_sub)(const char* endpoint, const char* filter, void* pipe);
    void* (*create_poller)(void* socket, int is_publisher); 
    void (*destroy)(void* socket);
    int (*send)(void* socket, const char* message);
    char* (*recv)(void* socket);
    int (*poll_socket)(void* poller, int is_publisher);
    void (*free)(const char* message);
} messaging_ops_t;

extern messaging_ops_t messaging_ops;

int cmpfunc(const void *a, const void *b);
void publisher(void *pipe, void *args);
void subscriber(void *pipe, void *args);

// -------------------------------
// ZeroMQ operations
// -------------------------------
void zmq_threads_init(int num_subs, subscriber_args_t subscribers[]);
void* zmq_create_pub(const char* endpoint, void* pipe);
void* zmq_create_sub(const char* endpoint, const char* filter, void* pipe);
void* zmq_create_poller(void* socket, int is_publisher);
void zmq_destroy(void* socket);
int zmq_send_msg(void* socket, const char* message);
char* zmq_recv_msg(void* socket);                   
int zmq_poll_socket(void* poller, int is_publisher);

// -------------------------------
// NanoMsg operations
// -------------------------------

void nn_threads_init(int num_subs, subscriber_args_t subscribers[]);
void* nn_create_pub(const char* endpoint, void* pipe);
void* nn_create_sub(const char* endpoint, const char* filter, void* pipe);
void* nn_create_poller(void* socket, int is_publisher);
void nn_destroy(void* socket);
int nn_send_msg(void* socket, const char* message); 
char* nn_recv_msg(void* socket);                    
int nn_poll_socket(void* poller, int is_publisher);

// -------------------------------
// NNG operations
// -------------------------------
void nng_threads_init(int num_subs, subscriber_args_t subscribers[]);
void* nng_create_pub(const char* endpoint, void* pipe);
void* nng_create_sub(const char* endpoint, const char* filter, void* pipe);
void* nng_create_poller(void* socket, int is_publisher);
void nng_destroy(void* socket);
int nng_send_msg(void* socket, const char* message); 
char* nng_recv_msg(void* socket);                    
int nng_poll_socket(void* poller, int is_publisher);


#endif // PUB_SUB_H
