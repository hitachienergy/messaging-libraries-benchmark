#include "pub-sub.h"
#include "utils.h"
#include <czmq.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"

/**
 * @brief Auxiliary function to destroy all the sub actors.
 */
void destroy_sub_actors(zactor_t *subs[], int num_subs) {
    for (int s = 0; s < num_subs; s++) {
        zactor_destroy(&subs[s]);
    }
}

/**
 * @brief Initializes and starts the ZeroMQ publisher and subscriber actors.
 */
void zmq_threads_init(int num_subs, subscriber_args_t subscribers[]) {
    zactor_t *subs[num_subs];
    for (int s = 0; s < num_subs; s++) {
        zactor_t *sub = zactor_new(subscriber, &subscribers[s]);
        if (!sub) {
            fprintf(stderr, "Failed to create ZeroMQ subscriber actor\n");
            return;
        }
        subs[s] = sub;
    }
    zactor_t *pub = zactor_new(publisher, NULL);
    if (!pub) {
        fprintf(stderr, "Failed to create ZeroMQ publisher actor\n");
        destroy_sub_actors(subs, num_subs);        
        return;
    }

    zactor_destroy(&pub);
    destroy_sub_actors(subs, num_subs);
}

#pragma GCC diagnostic pop

/**
 * @brief Creates a ZeroMQ publisher socket bound to the specified endpoint.
 * 
 * @param endpoint The endpoint to bind the publisher to.
 * @return void* Pointer to the publisher socket, or NULL on failure.
 */
void* zmq_create_pub(const char* endpoint, void* pipe) {
    zsock_t* pub;
    pub = zsock_new_pub(endpoint);

    if (!pub) {
        fprintf(stderr, "Failed to create ZeroMQ publisher socket\n");
    }
    if (pipe) zsock_signal(pipe, 0);
    return pub;
}

/**
 * @brief Creates a ZeroMQ subscriber socket connected to the specified endpoint with the given filter.
 * 
 * @param endpoint The endpoint to connect the subscriber to.
 * @param filter The subscription filter.
 * @return void* Pointer to the subscriber socket, or NULL on failure.
 */
void* zmq_create_sub(const char* endpoint, const char* filter, void* pipe) {
    zsock_t* sub;
    sub = zsock_new_sub(endpoint, filter);

    if (!sub) {
        fprintf(stderr, "Failed to create ZeroMQ subscriber socket\n");
        return NULL;
    }
    // since polling_enabled, the operations are non-blocking 
    zsock_set_rcvtimeo(sub, 0); // non-blocking
    
    if (pipe) zsock_signal(pipe, 0);
    return sub;
}


/**
 * @brief Creates a ZeroMQ poller for the specified socket.
 * 
 * @param socket The socket to create the poller for.
 * @param is_publisher 1 if the socket is a publisher, 0 if subscriber.
 * @return void* Pointer to the poller, or NULL on failure.
 */
void* zmq_create_poller(void* socket, int is_publisher) {
    void* raw_socket = zsock_resolve((zsock_t*)socket);
    if (!raw_socket) {
        fprintf(stderr, "Failed to resolve raw socket\n");
        return NULL;
    }

    zmq_pollitem_t* poll_item = malloc(sizeof(zmq_pollitem_t));
    if (!poll_item) {
        fprintf(stderr, "Failed to allocate memory for poller\n");
        return NULL;
    }
    poll_item->socket = raw_socket;
    poll_item->fd = 0;
    poll_item->revents = 0;
    poll_item->events = is_publisher ? ZMQ_POLLOUT : ZMQ_POLLIN;
    return poll_item;
}

/**
 * @brief Destroys a ZeroMQ socket.
 * 
 * @param socket Pointer to the socket to destroy.
 */
void zmq_destroy(void* socket) {
    if (socket) {
        zsock_destroy((zsock_t**)&socket);
    }
}

/**
 * @brief Polls a ZeroMQ socket for events.
 * 
 * @param poller The poller to use for polling.
 * @return int 0 on success, -1 on failure.
 */
int zmq_poll_socket(void* poller, int is_publisher) {
    zmq_pollitem_t* poll_item = (zmq_pollitem_t*)poller;
    int rc = zmq_poll(poll_item, 1, polling_timeout + pub_time_interval);
    if (rc < 0) {
        fprintf(stderr, "Error polling socket: %s\n", zmq_strerror(zmq_errno()));
        return -1;
    } else if (rc == 0) {
        return -1; // timeout occurred
    }
    int revents = poll_item->revents;
    if ((is_publisher && (revents & ZMQ_POLLOUT)) ||
        (!is_publisher && (revents & ZMQ_POLLIN))) {
        return 0; // Socket is ready
    }
    return -1; // Socket not ready
}

/**
 * @brief Frees a message allocated by the ZeroMQ library.
 * 
 * @param message The message to free.
 * 
 * @note zmq_free_msg is only used when zero-copy is not enabled, otherwise the messages will be freed through the zcp_free function.
 */
void zmq_free_msg(const char* message) {
    free((void*)message);
}
void zcp_free(void *data, void *hint) {
    (void)hint; // unused parameter
    free(data); // dealloc the message after sending
}

/**
 * @brief Sends a message over a ZeroMQ socket.
 * 
 * @param socket The socket to send the message through.
 * @param message The message to send.
 * @return int 0 on success, -1 on failure.
 */
int zmq_send_msg(void* socket, const char* message) {
    int rc = zstr_send((zsock_t*)socket, message);
    if (rc != 0) {
        fprintf(stderr, "Failed to send message: %s\n", zmq_strerror(zmq_errno()));
        return -1;
    }
    return 0;
}

/**
 * @brief Receives a message from a ZeroMQ socket.
 * 
 * @param socket The socket to receive the message from.
 * @return char* The received message, or NULL on failure.
 */
char* zmq_recv_msg(void* socket) {
    char* message = zstr_recv((zsock_t*)socket);
    if (!message) {
        fprintf(stderr, "Failed to receive message: %s\n", zmq_strerror(zmq_errno()));
    }
    return message;
}


messaging_ops_t ops_zeromq = {
    .threads_init = zmq_threads_init,
    .create_pub = zmq_create_pub,
    .create_sub = zmq_create_sub,
    .create_poller = zmq_create_poller,
    .destroy = zmq_destroy,
    .send = zmq_send_msg,
    .recv = zmq_recv_msg,
    .poll_socket = zmq_poll_socket,
    .free = zmq_free_msg
};
