#include "pub-sub.h"
#include "utils.h"
#include <nanomsg/nn.h>
#include <nanomsg/pubsub.h>
#include <nanomsg/pair.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

extern int subscriber_ready;
extern int publisher_ready;
extern int payload_length; // get payload length from user input

typedef struct {
    int sockfd;
} nn_socket_t;

/**
 * @brief Auxiliary function for subscriber thread.
 */
static void* subscriber_thread(void* arg) {
    subscriber(NULL, arg);
    return NULL;
}
/**
 * @brief Auxiliary function for publisher thread.
 */
static void* publisher_thread(void* arg) {
    (void)arg; // unused parameter
    publisher(NULL, NULL);
    return NULL;
}

/**
 * @brief Initializes and starts the NanoMsg publisher and subscriber threads.
 * pragma suppress warning about incompatible function pointer types
 */
void nn_threads_init(int num_subs, subscriber_args_t subscribers[]) {
    pthread_t sub_threads[num_subs];
    pthread_t pub_thread;

    for (int s=0; s < num_subs; s++) {
        pthread_t sub_thread;
        if (pthread_create(&sub_thread, NULL, subscriber_thread, &subscribers[s]) != 0) {
            fprintf(stderr, "Failed to create subscriber thread\n");
            return;
        }
        sub_threads[s] = sub_thread;
    }
    if (pthread_create(&pub_thread, NULL, publisher_thread, NULL) != 0) {
        fprintf(stderr, "Failed to create publisher thread\n");
        return;
    }
    for (int s=0; s < num_subs; s++) {
        pthread_join(sub_threads[s], NULL);
    }
    pthread_join(pub_thread, NULL);
}

/**
 * @brief Creates a NanoMsg publisher socket bound to the specified endpoint.
 * 
 * @param endpoint The endpoint to bind the publisher to.
 * @return void* Pointer to the publisher socket, or NULL on failure.
 */
void* nn_create_pub(const char* endpoint, void* pipe) {
    (void)pipe; // unused parameter

    int sock = nn_socket(AF_SP, NN_PUB);
    if (sock < 0) {
        fprintf(stderr, "Failed to create nanomsg publisher socket: %s\n", nn_strerror(nn_errno()));
        return NULL;
    }
    if (nn_bind(sock, endpoint) < 0) {
        fprintf(stderr, "Failed to bind nanomsg publisher socket to endpoint: %s, error: %s\n", endpoint, nn_strerror(nn_errno()));
        nn_close(sock);
        return NULL;
    }

    nn_socket_t* nn_sock = (nn_socket_t*)malloc(sizeof(nn_socket_t));
    if (!nn_sock) {
        fprintf(stderr, "Failed to allocate memory for nn_socket_t\n");
        nn_close(sock);
        return NULL;
    }
    nn_sock->sockfd = sock;
    return (void*)nn_sock;
}

/**
 * @brief Creates a NanoMsg subscriber socket bound to the specified endpoint with the given filter.
 * 
 * @param endpoint The endpoint to connect the subscriber to.
 * @param filter The subscription filter.
 * @return void* Pointer to the subscriber socket, or NULL on failure.
 */
void* nn_create_sub(const char* endpoint, const char* filter, void* pipe) {
    (void)pipe; // unused parameter

    int sock = nn_socket(AF_SP, NN_SUB);
    if (sock < 0) {
        fprintf(stderr, "Failed to create nanomsg subscriber socket: %s\n", nn_strerror(nn_errno()));
        return NULL;
    }
    if (nn_setsockopt(sock, NN_SUB, NN_SUB_SUBSCRIBE, filter, strlen(filter)) < 0) {
        fprintf(stderr, "Failed to set nanomsg subscriber filter: %s\n", nn_strerror(nn_errno()));
        nn_close(sock);
        return NULL;
    }
    if (nn_connect(sock, endpoint) < 0) {
        fprintf(stderr, "Failed to connect nanomsg subscriber socket to endpoint: %s, error: %s\n", endpoint, nn_strerror(nn_errno()));
        nn_close(sock);
        return NULL;
    }

    // since polling_enabled, the operations are non-blocking 
    int timeout = 0;
    nn_setsockopt(sock, NN_SOL_SOCKET, NN_RCVTIMEO, &timeout, sizeof(timeout));
    
    nn_socket_t* nn_sock = (nn_socket_t*)malloc(sizeof(nn_socket_t));
    if (!nn_sock) {
        fprintf(stderr, "Failed to allocate memory for nn_socket_t\n");
        nn_close(sock);
        return NULL;
    }
    nn_sock->sockfd = sock;
    return (void*)nn_sock;
}

/**
 * @brief Creates a NanoMsg poller for the specified socket.
 * 
 * @param socket The socket to create the poller for.
 * @param is_publisher 1 if the socket is a publisher, 0 if subscriber.
 * @return void* Pointer to the poller, or NULL on failure.
 */
void* nn_create_poller(void* socket, int is_publisher) {
    struct nn_pollfd* pfd = malloc(sizeof(struct nn_pollfd));
    if (!pfd) {
        fprintf(stderr, "Failed to allocate memory for poller\n");
        return NULL;
    }
    nn_socket_t* nn_sock = (nn_socket_t*)socket;
    pfd->fd = nn_sock->sockfd;
    pfd->events = is_publisher ? NN_POLLOUT : NN_POLLIN;
    pfd->revents = 0;
    return pfd;
}

/**
 * @brief Destroys a NanoMsg socket.
 * 
 * @param socket Pointer to the socket to destroy.
 */
void nn_destroy(void* socket) {
    (void)socket;  // unused parameter
    nn_close(((nn_socket_t*)socket)->sockfd);
}

/**
 * @brief Sends a message over a NanoMsg socket.
 * 
 * @param socket The socket to send the message through.
 * @param message The message to send.
 * @return int 0 on success, -1 on failure.
 */
int nn_send_msg(void* socket, const char* message) {
    nn_socket_t* nn_sock = (nn_socket_t*)socket;

    int bytes = nn_send(nn_sock->sockfd, message, strlen(message), 0);
    if (bytes < 0) {
        fprintf(stderr, "Failed to send message: %s\n", nn_strerror(nn_errno()));
        return -1;
    }
    return 0;
}

/**
 * @brief Receives a message from a NanoMsg socket.
 * 
 * @param socket The socket to receive the message from.
 * @return char* The received message, or NULL on failure.
 */
char* nn_recv_msg(void* socket) {
    nn_socket_t* nn_sock = (nn_socket_t*)socket;
    char* buf = NULL;
    int bytes = nn_recv(nn_sock->sockfd, &buf, NN_MSG, 0);
    if (bytes >= 0) {
        char* message = strndup(buf, bytes);
        nn_freemsg(buf);
        return message;
    } else if (nn_errno() == EAGAIN) {
        fprintf(stderr, "No message available\n");
        return NULL;
    } else {
        fprintf(stderr, "Failed to receive message: %s\n", nn_strerror(nn_errno()));
        return NULL;
    }
}


/**
 * @brief Polls a NanoMsg socket for events.
 * 
 * @param poller The poller to use for polling.
 * @param is_publisher 1 if the socket is a publisher, 0 if subscriber.
 * @return int 0 on success, -1 on failure.
 */
int nn_poll_socket(void* poller, int is_publisher) {
    struct nn_pollfd* pfd = (struct nn_pollfd*)poller;

    int rc = nn_poll(pfd, 1, polling_timeout + pub_time_interval);
    if (rc < 0) {
        fprintf(stderr, "Error polling socket: %s\n", nn_strerror(nn_errno()));
        return -1;
    } else if (rc == 0) {
        return -1; // timeout occurred
    }
    short revents = pfd->revents;
    if ((is_publisher && (revents & NN_POLLOUT)) ||
        (!is_publisher && (revents & NN_POLLIN))) {
        return 0; // Socket is ready
    }
    return -1; // Not ready
}

/**
 * @brief Frees the memory allocated for a message.
 * 
 * @param message The message to free.
 */
void nn_free_msg(const char* message) {
    free((void*)message);
    // TODO: consider using nn_freemsg() instead
}


messaging_ops_t ops_nanomsg = {
    .threads_init = nn_threads_init,
    .create_pub = nn_create_pub,
    .create_sub = nn_create_sub,
    .create_poller = nn_create_poller,
    .destroy = nn_destroy,
    .send = nn_send_msg,
    .recv = nn_recv_msg,
    .poll_socket = nn_poll_socket,
    .free = nn_free_msg
};