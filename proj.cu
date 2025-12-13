#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>


/// For one block we compute the simulation of one quadruplet of rho, kappa, sigma and theta
__global__ void euler(float* result, float *rhos, float *kappas, float *thetas, float *sigmas, float sqrtdt){

    extern __shared__ float data[];
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    
    float S= 1.0;
    float v= 0.1;
    float kappa= kappas[blockIdx.x];
    float sigma= sigmas[blockIdx.x];
    float theta= thetas[blockIdx.x];
    float rho= rhos[blockIdx.x];
    float nvar = sqrtf(1.f - rho*rho);
    float dt= sqrtdt * sqrtdt;

    curandState localState;
    curand_init(42, idx, 0, &localState);
    for (float i=0; i<1; i+= dt){

        float g1= curand_normal(&localState);
        float g2= curand_normal(&localState);

        float sqrv = sqrtf(v);
        S= S + sqrv * S * sqrtdt * (rho * g1 + nvar * g2);
        v= fabsf(v + kappa * (theta - v) * sqrtdt *sqrtdt + sigma * sqrv * sqrtdt * g1);
    }
    float res = fmaxf(S - 1.0, 0.0);

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
        result[blockIdx.x]= data[0] / blockDim.x;
    }
}

/// Implementation of Gamma distribution as described in https://dl.acm.org/doi/pdf/10.1145/358407.358414
__device__ float gamma( curandState *state, float a )
{
    if (a <= 0.0f) return 0.0f;

    float boost = 1;
    if (a < 1){
        boost= curand_uniform(state);
        
        boost= powf(boost, 1.0/a);
        a += 1;
    }

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
            return d*v*boost;
        if ( log( u ) < 0.5f*x*x + d*(1.f - v + log( v )) )
            return d*v*boost;
    }
}


__global__ void testGamma(float a, float* result )
{
    int lim = blockDim.x * gridDim.x;
    int idx0= blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState;
    curand_init(42, idx0, 0, &localState);
    for ( int k = 0; k < 1000; ++k )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x + k*lim;
        result[idx] = gamma( &localState, a);
    }
}

__global__ void almost(float* result, float *rhos, float* kappas, float* thetas, float* sigmas, float sqrtdt){

    extern __shared__ float data[];
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    
    float kappa= kappas[blockIdx.x];
    float sigma= sigmas[blockIdx.x];
    float theta= thetas[blockIdx.x];
    float rho = rhos[blockIdx.x];
    float nvar = sqrtf(1.f - rho*rho);
    
    float logS= 0.0;
    float v= 0.1;
    float sigma_square= sigma * sigma;
    float dt= sqrtdt * sqrtdt;
    float exp_k= expf(-kappa * dt);
    float d= 2 * kappa * theta / sigma_square;
    float k0= -rho * kappa * theta * dt / sigma;
    float k2= rho / sigma;
    float k1= (rho * kappa / sigma - 0.5) * dt - k2;

    curandState localState;
    curand_init(42, idx, 0, &localState);
    
    for (float t= 0; t< 1; t+= dt){
        float g1 = curand_normal(&localState);
        float g2 = curand_normal(&localState);

        float lambda= 2* kappa * exp_k * v / (sigma_square * (1 - exp_k));
        float N= curand_poisson(&localState, lambda);

        float vdt = sigma_square * (1 - exp_k) * gamma(&localState, d + N) / (2 * kappa);
        
        logS += k0 + k1 * v + k2 * vdt + nvar * sqrtf(v) * sqrtdt * (rho * g1 + nvar * g2);
        v = vdt;
    }
    
    float S1= expf(logS);
    float res = fmaxf(S1 - 1.0, 0.0);
    
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
        result[blockIdx.x]= data[0] / blockDim.x;
    }
}


struct CudaData
{
    float *kappa = nullptr, *theta = nullptr, *sigma = nullptr, *rho = nullptr;

    CudaData( const std::vector<float>& kappas, const std::vector<float>& sigmas, const std::vector<float>& thetas, const std::vector<float>& rhos )
    {
        auto size = kappas.size() * sizeof( float );
        assert( kappas.size() == sigmas.size() && sigmas.size() == thetas.size() && rhos.size() == rhos.size() );

        cudaMalloc(&kappa, size);
        cudaMalloc(&sigma, size);
        cudaMalloc(&theta, size);
        cudaMalloc(&rho, size);

        cudaMemcpy(kappa, kappas.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(sigma, sigmas.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(theta, thetas.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(rho, rhos.data(), size, cudaMemcpyHostToDevice);
    }

    ~CudaData()
    {
        cudaFree(kappa);
        cudaFree(theta);
        cudaFree(sigma);
        cudaFree(rho);
    }
};

float mse( const std::vector<float> &x, const std::vector<float>& y )
{
    assert( x.size() == y.size() );
    float res = 0.f;
    for (int i = 0; i < x.size(); i++)
        res += ( x[i] - y[i] )*( x[i] - y[i] );
    res = res / x.size();
    return res;
}

std::pair<float, float> meanvar( const std::vector<float> &x )
{
    float mean = 0, mean2 = 0;
    for ( float v : x )
    {
        mean += v;
        mean2 += v*v;
    }
    mean /= x.size();
    mean2 /= x.size();

    return {mean, mean2 - mean*mean};
}



void ex1()
{
    float dt= 1.0/1000.0;
    float sqrtdt= sqrtf(dt);
    float kappa= 0.5;
    float theta= 0.1;
    float sigma= 0.3;

    int NTPB = 1024;
    int NB = 10;        // for this exercise the number of blocks is the same as the number of values for rho
    size_t shared = NTPB * sizeof(float);

    std::vector<float> kappas( NB, kappa );
    std::vector<float> sigmas( NB, sigma );
    std::vector<float> thetas( NB, theta );
    std::vector<float> rhos;
    for ( int irho = 1; irho < NB + 1; ++irho )
        rhos.push_back((float)irho / (float)NB);

    CudaData cudaData( kappas, sigmas, thetas, rhos );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* gpu_result;
    cudaMalloc(&gpu_result, NB*sizeof(float));
    cudaMemset(gpu_result, 0, NB*sizeof(float));
    cudaEventRecord(start);
    euler<<<NB, NTPB, shared>>>(gpu_result, cudaData.rho, cudaData.kappa, cudaData.theta, cudaData.sigma, sqrtdt);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed= 0;
    cudaEventElapsedTime(&elapsed, start, stop);

    std::vector<float> result( NB );
    cudaMemcpy(result.data(), gpu_result, NB*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree( gpu_result );

    printf("[Exercise 1] elapsed time: %fms\n", elapsed);
    for ( int i = 0; i < NB; ++i )
        printf("[Exercise 1] for a value rho of %f we get a result E[(S-1)+]: %f\n", rhos[i], result[i]);
}

void ex3()
{
    float dt= 1.0/1000.0;
    float sqrtdt= sqrtf(dt);
    std::vector<float> kappas;
    std::vector<float> sigmas;
    std::vector<float> thetas;
    std::vector<float> rhos;
    for (float rho = 0.1; rho < 1; rho += 0.1 ){
        for (float kappa= 0.1; kappa<10; kappa += 0.5){
            for (float sigma= 0.1; sigma< 1; sigma += 0.05){
                for (float theta= 0.01; theta< 0.5; theta += 0.025){
                    if (20 * kappa * theta > sigma * sigma){
                        kappas.push_back(kappa);
                        sigmas.push_back(sigma);
                        thetas.push_back(theta);
                        rhos.push_back(rho);
                    }
                }
            }
        }
    }
    int NTPB = 1024;
    int NB = kappas.size();   // for this exercise, each combination of parameters corresponds to a block
    size_t shared = NTPB * sizeof(float);

    CudaData cudaData( kappas, sigmas, thetas, rhos );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* gpu_result;
    cudaMalloc(&gpu_result, NB*sizeof(float));
    cudaMemset(gpu_result, 0, NB*sizeof(float));
    cudaEventRecord(start);
    euler<<<NB, NTPB, shared>>>(gpu_result, cudaData.rho, cudaData.kappa, cudaData.theta, cudaData.sigma, sqrtdt);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float eulerElapsed= 0;
    cudaEventElapsedTime(&eulerElapsed, start, stop);
    cudaDeviceSynchronize();

    std::vector<float> eulerResult( NB );
    cudaMemcpy(eulerResult.data(), gpu_result, NB*sizeof(float), cudaMemcpyDeviceToHost);


    cudaMemset(gpu_result, 0, NB*sizeof(float));
    cudaEventRecord(start);
    almost<<<NB, NTPB, shared>>>(gpu_result, cudaData.rho, cudaData.kappa, cudaData.theta, cudaData.sigma, sqrtdt);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float almostElapsed= 0;
    cudaEventElapsedTime(&almostElapsed, start, stop);
    cudaDeviceSynchronize();

    std::vector<float> almostResult( NB );
    cudaMemcpy(almostResult.data(), gpu_result, NB*sizeof(float), cudaMemcpyDeviceToHost);
    auto [almostMean, almostVar] = meanvar( almostResult );

    printf("[Exercise 3] elapsed time for Euler method: %fms\n", eulerElapsed);
    printf("[Exercise 3] elapsed time for almost exact method: %fms\n", almostElapsed);
    printf("[Exercise 3] mean squared distance between two methods: %f (vs mean value %f and variance %f)\n", mse( eulerResult, almostResult ), almostMean, almostVar);


    // now change dt and redo
    dt = 1.f/30.f;
    sqrtdt = sqrtf( dt );

    cudaMemset(gpu_result, 0, NB*sizeof(float));
    cudaEventRecord(start);
    almost<<<NB, NTPB, shared>>>(gpu_result, cudaData.rho, cudaData.kappa, cudaData.theta, cudaData.sigma, sqrtdt);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float almost2Elapsed= 0;
    cudaEventElapsedTime(&almost2Elapsed, start, stop);
    cudaDeviceSynchronize();

    std::vector<float> almost2Result( NB );
    cudaMemcpy(almost2Result.data(), gpu_result, NB*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree( gpu_result );

    printf("[Exercise 3] elapsed time for almost exact method with dt = 1/30: %fms\n", almost2Elapsed);
    printf("[Exercise 3] mean squared distance between almost exact method with different dt: %f\n", mse(almostResult, almost2Result));
}


int main(void)
{
    ex1();
    ex3();

//##############uncomment to verify Gamma distribution############
//     int NTPB = 512;
//     int NB = 128;
//     int tot= NTPB * NB;
//     cudaMalloc(&rv1, 1000 * tot * sizeof(curandState));
    
//     float *distrGPU, *distrCPU;
    
//    cudaMalloc( &distrGPU, 1000*tot*sizeof(float) );
//    distrCPU = (float*)malloc( 1000*tot*sizeof(float) );
//    testGamma<<<NB, NTPB>>>( 0.5, distrGPU );
//    cudaMemcpy( distrCPU, distrGPU, 1000*tot*sizeof(float), cudaMemcpyDeviceToHost );

//    FILE* f = fopen( "/tmp/distr.csv", "w" );
//    assert(f);
//    for ( size_t i = 0; i < 1000*tot; ++i )
//    {
//        fprintf( f, "%f\n", distrCPU[i] );
//    }
//    fclose( f );

    return 0;
}

