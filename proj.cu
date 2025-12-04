#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
// rho belongs to [-0.9, 0.9]

__global__ void init_normal(curandState *states){
    int lim = blockDim.x * gridDim.x;
    for (int k=0; k<1000; k++){
        int idx= blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(0,idx + k * lim,0,&states[idx + k * lim]);
    }
}

__device__ float gamma( curandState *state, float a )
{
    float d = a - 1.f/3.f;
    float c = 1.f / sqrt(9.f*d);
    float x, v, u;
    while( true )
    {
        do {
            x = curand_normal( state );
            v = 1.f + c*x;
        }
        while ( v <= 0.f );

        v = v*v*v;
        u = curand_uniform( state );
        if ( u < 1.f - 0.0331*x*x*x*x )
            return d*v;
        if ( log( u ) < 0.5f*x*x + d*(1.f - v + log( v )) )
            return d*v;
    }
}

__global__ void testGamma( curandState* states, float a, float* result )
{
    int lim = blockDim.x * gridDim.x;
    for ( int k = 0; k < 1000; ++k )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x + k*lim;
        result[idx] = gamma( &states[idx], a );
    }
}

__global__ void Euler(curandState* G1, curandState* G2, float rho, float* result, float kappa, float theta, float sigma, float sqrtdt, float nvar){

    extern __shared__ float data[];
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    int lim= blockDim.x * gridDim.x;
    

    float S= 1.0;
    float v= 0.1;
    for (int i=0; i<1000; i++){
        curandState val1= G1[idx + i * lim];
        curandState val2= G2[idx + i * lim];

        float g1= curand_normal(&val1);
        float g2= curand_normal(&val2);

        float sqrv = sqrtf(v);
        S= S + sqrv * S * sqrtdt * (rho * g1 + nvar * g2);
        v= fabsf(v + kappa * (theta - v) * sqrtdt *sqrtdt + sigma * sqrv * sqrtdt * g1);
    }
    float res = max(S - 1.0, 0.0);

    data[threadIdx.x] = res;
    __syncthreads();

    int k= blockDim.x / 2;
    while (k!=0){
        if (threadIdx.x < k){
            data[threadIdx.x] += data[threadIdx.x + k];
        }
        __syncthreads();
        k /= 2;
    }
    if (threadIdx.x == 0){
        atomicAdd(result, data[0]);
    }
}




int main(void){
    float kappa= 0.5;
    float theta= 0.1;
    float sigma= 0.3;
    float dt= 0.001;
    float sqrtdt= sqrtf(dt);
    
    // we will make one rho per thread and one G1, G2 per block or invert
    int NTPB = 512;
    int NB = 128;
    int tot= NTPB * NB;
    curandState *rv1;
    curandState *rv2;
    cudaMalloc(&rv1, 1000 * tot * sizeof(curandState));
    cudaMalloc(&rv2, 1000 * tot * sizeof(curandState));
    size_t shared= NTPB * sizeof(float);
    init_normal<<<NB, NTPB>>>(rv1);
    init_normal<<<NB, NTPB>>>(rv2);

// uncomment to verify Gamma distribution
//    float *distrGPU, *distrCPU;
//    cudaMalloc( &distrGPU, 1000*tot*sizeof(float) );
//    distrCPU = (float*)malloc( 1000*tot*sizeof(float) );
//    testGamma<<<NB, NTPB>>>( rv1, 5, distrGPU );
//    cudaMemcpy( distrCPU, distrGPU, 1000*tot*sizeof(float), cudaMemcpyDeviceToHost );
//
//    FILE* f = fopen( "/tmp/distr.csv", "w" );
//    assert( f );
//    for ( size_t i = 0; i < 1000*tot; ++i )
//    {
//        fprintf( f, "%f\n", distrCPU[i] );
//    }
//    fclose( f );
    
    for (float rho= -0.9; rho<0.9; rho +=0.2){
        float nvar= sqrtf(1 - rho*rho);
        float *cusum;
        cudaMalloc(&cusum, sizeof(float));
        cudaMemset(cusum, 0, sizeof(float));
        Euler<<<NB, NTPB, shared>>>(rv1, rv2, rho, cusum, kappa, theta, sigma, sqrtdt, nvar);

        float sum;
        cudaMemcpy(&sum, cusum, sizeof(float), cudaMemcpyDeviceToHost);
        sum /= tot;

        printf("for a value rho of %f we get a result E[(S-1)+] %f\n", rho, sum);
        cudaFree(cusum);
     
    }
}