#include <stdio.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

void setColours(vector<set<int>>& adj, int maxDegree, int n, int m, vector<int> &colours, vector<bool> & conflicts){
    int maxColours = maxDegree + 1;
    int i, j;
    
    for(i = 0; i < n; i++){
        if(!conflicts[i])
            continue;
        vector<bool> no(maxColours + 1, false);
        for(auto j : adj[i])
            no[colours[j]] = true;
        for(j = 1; j <= maxColours; j++){
            if(!no[j]){
                colours[i] = j;
                break;
            }
        }
    }
}

bool checkConflicts(vector<set<int>>& adj, int maxDegree, int n, int m, vector<int> &colours, vector<bool>& conflicts){
    bool isConflict = false;
    int i;
    for(i = 0; i < n; i++){
        conflicts[i] = false;
        for(auto j : adj[i]){
            if(colours[j] == colours[i] && j < i){
                conflicts[i] = true;
                isConflict = true;
            }
        }
    }
    return isConflict;
}

void readGraph(int &n, int &m, int &maxDegree, int** h_degree, int** h_adj, int** h_adj_p){     
    int i, k, c;
    cout << "Enter the number of vertices : " << endl;
    cin >> n;
    cout << "Enter the number of edges : " << endl;
    cin >> m;
    if(m > (((long long)n * (n - 1))/ 2)){
        cout << "Invalid number of edges." << endl;
        exit(0);
        return;
    }

    cout << "Random graph or manual? Enter 0 or 1" << endl;
    cin >> c;

    vector<set<int>> adj(n);
    if(c == 0){                                                         //For random graphs
        i = 0;
        while(i < m){
            int x, y;
            do{
                x = rand() % n;
                y = rand() % n;
            }while(x == y);

            if(adj[x].find(y) != adj[x].end())
                continue;
            // printf("%d --- %d\n", x, y);
            adj[x].insert(y);
            adj[y].insert(x);
            i++;
        }
    }
    else{                                                              //For manually entered graphs
        i = 0;
        while(i < m){
            printf("Click 1 to enter edge and 0 to finish.\n");
            scanf("%d", &k);
            if(!k)
                break;
            int s, d;
            printf("Enter start and end of edge in 1-ordering : \n");
            scanf("%d %d", &s, &d);
            if(s == d){
                printf("Invalid edge.\n");
                continue;
            }
            if(s > n || s < 1 || d > n || d < 1){
                printf("Invalid edge.\n");
                continue;
            }
            adj[s - 1].insert(d - 1);
            adj[d - 1].insert(s - 1);
            i++;
        }
    }

    *h_adj_p = new int[n + 1];
    *h_adj = new int[(2 * m) + 1];
    *h_degree = new int[n];

    int point = 0;
    for(i = 0;i < n; i++){
        (*h_adj_p)[i] = point;
        for(auto j : adj[i])
            (*h_adj)[point++] = j;
    }
    (*h_adj_p)[n] = point;

    int mx = INT_MIN; 
    for(i = 0;i < n; i++){
		(*h_degree)[i] = (int)adj[i].size();                                                         
        mx = max(mx, (int)adj[i].size());
	}
    
    maxDegree = mx;

    //----------------------------------------Serial Graph Coloring---------------------------

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<bool> conflicts(n, true);
    vector<int> colours(n, 0);
    cudaEventRecord(start);
    
    do{
        setColours(adj, maxDegree, n, m, colours, conflicts);
    }while(checkConflicts(adj, maxDegree, n, m, colours, conflicts));


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The minimum colors needed => %d\n", *max_element(colours.begin(), colours.end()));
    printf("Serial graph coloring takes => %f ms\n",milliseconds);

    //----------------------------------------------------------------------------------------

}

__global__ void initializeRemCol(bool* d_rem, int* d_colours, int n){               //Set uncolored vertex set to true and initial colour to invalid(0)
    int ind = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(ind >= n)
        return;
    d_rem[ind] = true;
    d_colours[ind] = 0;
}

__global__ void colorSet(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree){
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(index >= n)
        return;
    
    if(!d_rem[index])                                                   //If current vertex is already colored, return
        return;

    int i, j, maxColours = maxDegree + 1;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Check if any uncolored neighbour has higher weight        
        j = d_adj[i];                                                                   
        if(d_rem[j] && d_weights[j] > d_weights[index])
            return;
    }

    d_rem[index] = false;                                               //Current vertex should be coloured so remove it from remaining set

    bool* forbidden = new bool[maxColours + 1];
    for(i = 0; i < maxColours + 1; i++)
        forbidden[i] = false;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Find out neighbours colors
        j = d_adj[i];
        if(!d_rem[j]) forbidden[d_colours[j]] = true;
    }
    for(i = 1; i <= maxColours; i++){                                   //Assign color which is not there in neighbours' colors
        if(!forbidden[i]){
            d_colours[index] = i;
            delete [] forbidden;
            return;
        }
    }
    delete [] forbidden;
}

int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights){
    int i, rem_size = n, mx = min(n, 1024);                                               //Initialize all variables
    bool* h_rem = new bool[n];
    int* h_colours = new int[n];
    bool* d_rem;
    int *d_colours;

    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        //Allocate space on GPU
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);
	
	int blocks = (n + mx - 1)/mx;
    initializeRemCol<<<blocks, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        colorSet<<<blocks, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree);    //Launch kernel
        cudaMemcpy(h_rem, d_rem, sizeof(bool) * n, cudaMemcpyDeviceToHost);                 //Copy back the updated uncolored set
        int k = 0;
        for(i = 0; i < n; i++){
            if(h_rem[i])
                k++;
        }
        rem_size = k;
    }

    cudaMemcpy(h_colours, d_colours, sizeof(int) * n, cudaMemcpyDeviceToHost);
    
    cudaFree(d_colours);                                                                    //Free memory
    cudaFree(d_rem);

    return h_colours;
}


__global__ void colorSet2(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree, int* d_degree){//pass h_deg here
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(index >= n)
        return;
    
    if(!d_rem[index])                                                   //If current vertex is already colored, return
        return;

    int i, j, maxColours = maxDegree + 1;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Check if any uncolored neighbour has higher weight        
        j = d_adj[i];                                                                   
        if(d_rem[j] && (d_degree[index]<d_degree[j]  || (d_degree[index]==d_degree[j] && d_weights[j] > d_weights[index])) )  
            return;
    }

    d_rem[index] = false;                                               //Current vertex should be coloured so remove it from remaining set

    bool* forbidden = new bool[maxColours + 1];
    for(i = 0; i < maxColours + 1; i++)
        forbidden[i] = false;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Find out neighbours colors
        j = d_adj[i];
        forbidden[d_colours[j]] = true;
    }
    for(i = 1; i <= maxColours; i++){                                   //Assign color which is not there in neighbours' colors
        if(!forbidden[i]){
            d_colours[index] = i;
            delete [] forbidden;
            return;
        }
    }
    delete [] forbidden;
}

int* colourGraph2(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights, int* h_degree){
    int i, rem_size = n, mx = min(n, 1024);                                               //Initialize all variables
    bool* h_rem = new bool[n];
    bool* h_inc = new bool[n];
    int* h_colours = new int[n];
    bool* d_rem;
    int *d_colours;
    int *d_degree;

    cudaMalloc((void**)&d_degree, sizeof(int) * n); 
    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        //Allocate space on GPU
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);

    cudaMemcpy(d_degree, h_degree, sizeof(int) * n, cudaMemcpyHostToDevice);
	
	int blocks = (n + mx - 1)/mx;
    initializeRemCol<<<blocks, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        // cout << rem_size << endl;
        colorSet2<<<blocks, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree, d_degree);    //Launch kernel
        cudaMemcpy(h_rem, d_rem, sizeof(bool) * n, cudaMemcpyDeviceToHost);                 //Copy back the updated uncolored set
        int k = 0;
        for(i = 0; i < n; i++){
            if(h_rem[i])
                k++;
        }
        rem_size = k;
    }

    cudaMemcpy(h_colours, d_colours, sizeof(int) * n, cudaMemcpyDeviceToHost);
    
    cudaFree(d_colours);                                                                    //Free memory
    cudaFree(d_rem);
    cudaFree(d_degree);

    return h_colours;
}

void randomWeightAssign(int* h_weights, int n){                                             //Assign random weights to the vertices
    vector<int> arr(n);
    for(int i=0; i<n; i++) arr[i] = i;
    random_shuffle(arr.begin(), arr.end());

    for(int i=0; i<n; i++)
        h_weights[i] = arr[i];
}

int main(){
    int n, m, maxDegree;
    srand(time(0));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* h_adj = NULL, *h_adj_p = NULL;
    int* d_adj, *d_adj_p;
    int* h_degree = NULL;

    readGraph(n, m, maxDegree, &h_degree, &h_adj, &h_adj_p);                                       //Take input
    int* h_weights = new int[n];
    int* d_weights;

    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));                            //Allocate space on GPU
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    cudaMalloc((void**)&d_weights, sizeof(int) * n);

	//----------------------------------------Jones Plassmann Graph Coloring--------------------------

    randomWeightAssign(h_weights, n);

    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);      //Copy data to GPU
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);

    cout << "The max degree is : " << maxDegree << endl;

    cudaEventRecord(start);
    
    int *colouring = colourGraph(n, m, maxDegree, d_adj, d_adj_p, d_weights);        
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int min_colors = INT_MIN;
    for(int i=0; i<n; i++)
        min_colors = max(min_colors, colouring[i]);

    printf("The minimum colors needed => %d\n", min_colors);
    printf("Parallel Graph coloring using Jones Plassmann algorithm takes => %f ms\n",milliseconds);
    
    //----------------------------------------Largest Degree First Graph Coloring--------------------------
    
    cudaEventRecord(start);
    
    int *colouring2 = colourGraph2(n, m, maxDegree, d_adj, d_adj_p, d_weights, h_degree);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    min_colors = INT_MIN;
    for(int i=0; i<n; i++)
        min_colors = max(min_colors, colouring2[i]);

    printf("The minimum colors needed => %d\n", min_colors);
    printf("Parallel Graph coloring using Largest degree first takes => %f ms\n",milliseconds); 
    cudaFree(d_adj_p);
    cudaFree(d_adj);    
    cudaFree(d_weights);    
}

