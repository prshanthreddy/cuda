#include <stdio.h>

int main(){
    int value=32;
    int *p=&value;
    int **q=&p;
    int ***r=&q;
    printf("Value of value: %d\n", value);
    printf("Address of value: %p\n", &value);
    printf("Value : %d\n", ***r);
    
    return 0;

}