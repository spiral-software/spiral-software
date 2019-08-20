/*
** iface_tcpip.c -- a stream socket client demo
*/
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>

#ifndef WIN32
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "iface.h"

#define BUF_SIZE 4096
#define PORT 3490                    /* the port client will be connecting to */
#define MAXDATASIZE 4096              /* max number of bytes we can get at once */
#define RSERVER "10.10.10.44"        /* address of remote server */
#define MACHINE_NAME "demo_client1"  /* name to register this client on server */

int connected_sockfd;
int ConnectToServer(char *client_name);

int tcpip_main(){

  int  exec_status;
  char input[BUF_SIZE];
  char output[BUF_SIZE];
  int  numbytes;  

  if(ConnectToServer(MACHINE_NAME) != 0){
    printf("Network Error");
    return INTER_EXIT;
  }
  
  exec_status = 1;
  
  while(exec_status != EXEC_QUIT){
    printf("\nSpiral Network Mode: Waiting for command:");

    if ((numbytes=recv(connected_sockfd, input, MAXDATASIZE-1, 0)) == -1) {
      perror("recv");
      return -1;
    }
    input[numbytes] = '\0';
    
    printf(" %s\n", input);

    exec_status = execute(input, output);
    
    if (send(connected_sockfd, output, strlen(output)+1, 0) == -1){
      perror("finish call failed");
    }
  }

  close(connected_sockfd);
  return INTER_EXIT;
}

void tcpip_printf(char *data){
  printf("tcpip_callback: %s", data);
}

int ConnectToServer(char *client_name)
{
    int sockfd, numbytes;  
    char buf[MAXDATASIZE];
    struct hostent *he;
    struct sockaddr_in their_addr; // connector's address information 

    if ((he=gethostbyname(RSERVER)) == NULL) {  // get the host info 
        perror("gethostbyname");
        return -1;
    }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        return -1;
    }

    their_addr.sin_family = AF_INET;    // host byte order 
    their_addr.sin_port = htons(PORT);  // short, network byte order 
    their_addr.sin_addr = *((struct in_addr *)he->h_addr);
    memset(&(their_addr.sin_zero), '\0', 8);  // zero the rest of the struct 

    if (connect(sockfd, (struct sockaddr *)&their_addr, sizeof(struct sockaddr)) == -1) {
        perror("connect");
        return -1;
    }
    
    if (send(sockfd, client_name, strlen(client_name) + 1, 0) == -1){
      perror("client_register failed");
      return -1;
    }
    
    if ((numbytes=recv(sockfd, buf, MAXDATASIZE-1, 0)) == -1) {
        perror("recv");
        return -1;
    }

    buf[numbytes] = '\0';

    printf("Received: %s",buf);

    connected_sockfd = sockfd;
    
    return 0;
}
