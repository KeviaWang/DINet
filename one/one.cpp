//
//  Created by HISS on 2020/11/5.
//  Copyright ? 2020 HISS. All rights reserved.
//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <iomanip>
#include <time.h>
#include <pthread.h>

using namespace std;

int n;
int thread_count;
double sum = 0.0;
pthread_mutex_t mutex;

void* Thread_sum(void* rank)
{
    int my_rank = *(int *) rank;
    double my_num = 0.0;
    int i;
    int my_n = n/thread_count;
    int my_first_i = my_n * my_rank;
    int my_last_i = my_first_i + my_n;
    
    for(i = my_first_i; i < my_last_i; i ++)
    {
        srand(i);
        double x = ((double)(rand() % n))/n;
        double y = ((double)(rand() % n))/n;
        cout<<x<<" "<<y<<endl; 
        if(x*x + y*y <= 1)
            my_num ++;
    }
    
    pthread_mutex_lock(&mutex);
    sum += my_num;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
    /* Thread_sum */
}

int main(void)
{
    struct timeval time1, time2;
    cout << "输入n值：" << endl;
    cin >> n ;
    cout << "输入线程数：" << endl;
    cin >> thread_count ;
    
    gettimeofday(&time1, NULL);
    pthread_t thread_ID[thread_count];
    
    int value[thread_count];
    for(int i = 0; i < thread_count; i ++)
        value[i] = i;
    
    //Create the thread, passing &value for the argument.
    for(int i = 0; i < thread_count; i++)
        pthread_create(&thread_ID[i], NULL, Thread_sum, &value[i]);
    
    //Wait for the thread to terminate.
    for(int i = 0; i < thread_count; i ++)
        pthread_join(thread_ID[i], NULL);
    
    sum = 4*sum/n;
    printf("%.20lf\n", sum);
    gettimeofday(&time2, NULL);
    printf("s: %ld, ms: %ld\n", time2.tv_sec-time1.tv_sec, (time2.tv_sec*1000 + time2.tv_sec/1000)-(time1.tv_sec*1000 + time1.tv_sec/1000));
    
    return 0;
}

