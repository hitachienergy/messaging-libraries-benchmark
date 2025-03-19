#include "pub-sub.h"
#include "utils.h"
#include <poll.h> 
#include <nng/nng.h>
#include <nng/protocol/pubsub0/pub.h>
#include <nng/protocol/pubsub0/sub.h>

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
 * @brief Initializes and starts the NNG publisher and subscriber threads.
 * pragma suppress warning about incompatible function pointer types
 */
void nng_threads_init(int num_subs, subscriber_args_t subscribers[]) {
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
 * @brief Auxiliary function to create a pointer to the socket to return.
 */
void* nng_allocate_socket(nng_socket sock) {
    nng_socket* sock_ptr = malloc(sizeof(nng_socket));
    if (sock_ptr == NULL) {
        fprintf(stderr, "Failed to allocate memory for NNG socket\n");
        nng_close(sock);
        return NULL;
    }
    *sock_ptr = sock;
    return (void*)sock_ptr;
}

/**
 * @brief Creates a NNG publisher socket bound to the specified endpoint.
 * 
 * @param endpoint The endpoint to bind the publisher to.
 * @return void* Pointer to the publisher socket, or NULL on failure.
 */
void* nng_create_pub(const char* endpoint, void* pipe) {
    (void)pipe; // unused parameter
    nng_socket pub;
    int rv;

    if (nng_pub0_open(&pub) != 0) {
        fprintf(stderr, "Failed to create NNG publisher socket\n");
        return NULL;
    }
    
    if ((rv = nng_listen(pub, endpoint, NULL, 0)) != 0) {
        fprintf(stderr, "Failed to bind NNG publisher socket to endpoint %s: %s\n", endpoint, nng_strerror(rv));
        nng_close(pub);
        return NULL;
    }

    int pub_fd;
    if (nng_socket_get_int(pub, NNG_OPT_SENDFD, &pub_fd) != 0) {
        fprintf(stderr, "Failed to get publisher socket fd\n");
    }
    
    return nng_allocate_socket(pub); // return pointer to the socket
}


/**
 * @brief Creates a NNG subscriber socket connected to the specified endpoint with the given filter.
 * 
 * @param endpoint The endpoint to connect the subscriber to.
 * @param filter The subscription filter.
 * @return void* Pointer to the subscriber socket, or NULL on failure.
 */
void* nng_create_sub(const char* endpoint, const char* filter, void* pipe) {
    (void)pipe; // unused parameter
    int rv;
    nng_socket sub;

    if (nng_sub0_open(&sub) != 0) {
        fprintf(stderr, "Failed to create NNG subscriber socket\n");
        return NULL;
    }

    if (nng_socket_set(sub, NNG_OPT_SUB_SUBSCRIBE, filter, strlen(filter)) != 0) {
        fprintf(stderr, "Failed to set NNG subscriber filter\n");
        nng_close(sub);
        return NULL;
    }
    
    int retry_count = 1000; // retries
    int retry_delay = 1; // delay between retries (milliseconds)

    // since polling is disabled, all operations are blocking with timeout
    nng_socket_set_ms(sub, NNG_OPT_RECVTIMEO, (retry_count * retry_delay + pub_start_delay + polling_timeout));

    while (retry_count > 0) {
        if ((rv = nng_dial(sub, endpoint, NULL, 0)) == 0) {
            if (retry_count < 5) {
                fprintf(stderr, "Connected NNG subscriber socket to endpoint %s after %d retries\n", endpoint, 5 - retry_count);
            }
            break; // connected
        }
        retry_count--;
        nng_msleep(retry_delay); // wait before retrying
    }

    if (rv != 0) {
        fprintf(stderr, "Failed to connect NNG subscriber socket to endpoint %s after retries: %s\n", endpoint, nng_strerror(rv));
        nng_close(sub);
        return NULL;
    }

    int sub_fd;
    if (nng_socket_get_int(sub, NNG_OPT_RECVFD, &sub_fd) != 0) {
        fprintf(stderr, "Failed to get subscriber socket fd\n");
    }
    return nng_allocate_socket(sub);
}


/**
 * @brief Polling is not supported in NNG; this function is not implemented.
 *
 * @param socket The socket to create the poller for.
 * @param is_publisher 1 if the socket is a publisher, 0 if subscriber.
 * @return void* Returns NULL as polling is not supported.
 */
void* nng_create_poller(void* socket, int is_publisher) {
    (void)socket; // unused parameter
    (void)is_publisher; // unused parameter
    return NULL;
}


/**
 * @brief Destroys a NNG socket.
 * 
 * @param socket Pointer to the socket to destroy.
 */
void nng_destroy(void* socket) {
    nng_socket* sock = (nng_socket*)socket;
    if (sock) {
        nng_close(*sock);
        free(sock);
    }
}


/**
 * @brief Sends a message over a NNG socket.
 * 
 * @param socket The socket to send the message through.
 * @param message The message to send.
 * @return int 0 on success, -1 on failure.
 */
int nng_send_msg(void* socket, const char* message) {
    nng_socket* nng_sock = (nng_socket*)socket;
    int rv;
    if ((rv = nng_send(*nng_sock, (char*)message, strlen(message), 0)) != 0) {
        fprintf(stderr, "Failed to send message: %s\n", nng_strerror(rv));
        return -1;
    }
    return 0;
}

/** 
 * @brief Receives a message from a NNG socket.
 * 
 * @param socket The socket to receive the message from.
 * @return char* The received message, or NULL on failure.
 */
char* nng_recv_msg(void* socket) {
    nng_socket* nng_sock = (nng_socket*)socket;
    int rv;

    nng_msg* msg;
    if ((rv = nng_recvmsg(*nng_sock, &msg, 0)) != 0) {
        fprintf(stderr, "Failed to receive message: %s\n", nng_strerror(rv));
        return NULL;
    }
    char* result = strndup(nng_msg_body(msg), nng_msg_len(msg));
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate memory for NNG message body\n");
        nng_msg_free(msg);
        return NULL;
    }
    nng_msg_free(msg);
    return result;
}


/**
 * @brief Polling is not supported in NNG; this function is not implemented.
 *
 * @param poller The poller to use for polling.
 * @param is_publisher 1 if the socket is a publisher, 0 if subscriber.
 * @return int Returns -1 as polling is not supported.
 */
int nng_poll_socket(void* poller, int is_publisher) {
    (void)poller; // unused parameter
    (void)is_publisher; // unused parameter
    return -1;
}

/**
 * @brief Frees the memory allocated for a message.
 * 
 * @param message The message to free.
 */
void nng_free_msg(const char* message) {
    free((void*)message);
}


messaging_ops_t ops_nng = {
    .threads_init = nng_threads_init,
    .create_pub = nng_create_pub,
    .create_sub = nng_create_sub,
    .create_poller = nng_create_poller,
    .destroy = nng_destroy,
    .send = nng_send_msg,
    .recv = nng_recv_msg,
    .poll_socket = nng_poll_socket,
    .free = nng_free_msg
};
