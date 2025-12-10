#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
// rho belongs to [-0.9, 0.9]
// using std::vector;

// __global__ void init_prob(curandState *states){
//     int lim = blockDim.x * gridDim.x;
//     for (int k=0; k<1000; k++){ //for every arrengement we do 1000 simulations and we need one RV per unit of time
//         int idx= blockIdx.x * blockDim.x + threadIdx.x;
//         curand_init(0,idx + k * lim,0,&states[idx + k * lim]);
//     }
// }


/// for one block we compute the simulation of one triplet of kappa, sigma ant theta
__global__ void Euler(float rho, float* result, float *kappas, float *thetas, float *sigmas, float sqrtdt, float nvar){

    extern __shared__ float data[];
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    
    float S= 1.0;
    float v= 0.1;
    float kappa= kappas[blockIdx.x];
    float sigma= sigmas[blockIdx.x];
    float theta= thetas[blockIdx.x];
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
        result[blockIdx.x]= data[0];
    }
}

__device__ float gamma( curandState *state, float a )
{
    if (a <= 0.0f) return 0.0f;

    float boost = 1;
    if (a < 1){
        boost= curand_uniform(state);
        a += 1;
        boost= powf(boost, 1.0/a);
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

__global__ void almost(float rho, float* result, float* kappas, float* thetas, float* sigmas, float sqrtdt, float nvar){

    extern __shared__ float data[];
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    
    float kappa= kappas[blockIdx.x];
    float sigma= sigmas[blockIdx.x];
    float theta= thetas[blockIdx.x];
    
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
        result[blockIdx.x]= data[0];
    }
}



int main(void){
    
    float dt= 1.0/1000.0;
    float sqrtdt= sqrtf(dt);

//#########uncomment for question 1##########################

    // float kappa= 0.5;
    // float theta= 0.1;
    // float sigma= 0.3;


    // // we will make one rho per thread and one G1, G2 per block or invert
    // int NTPB = 1024;
    // int NB = 1;
    // int tot= NTPB * NB;

    // std::vector<float> kappas;
    // std::vector<float> sigmas;
    // std::vector<float> thetas;

    // kappas.push_back(kappa);
    // sigmas.push_back(sigma);
    // thetas.push_back(theta);

    // float *gpu_kappa, *gpu_theta, *gpu_sigma;

    // cudaMalloc(&gpu_kappa, sizeof(float));
    // cudaMalloc(&gpu_sigma, sizeof(float));
    // cudaMalloc(&gpu_theta, sizeof(float));

    // cudaMemcpy(gpu_kappa, kappas.data(),
    //         sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemcpy(gpu_sigma, sigmas.data(),
    //         sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemcpy(gpu_theta, thetas.data(),
    //         sizeof(float), cudaMemcpyHostToDevice);


    
    // size_t shared= NTPB * sizeof(float);
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // for (float rho= -0.9; rho<0.9; rho +=0.2){
    //     float nvar= sqrtf(1 - rho*rho);
    //     float *cusum;
    //     cudaMalloc(&cusum, sizeof(float));
    //     cudaMemset(cusum, 0, sizeof(float));

    //     cudaEventRecord(start);
    //     // Euler<<<NB, NTPB, shared>>>(rho, cusum, gpu_kappa, gpu_theta, gpu_sigma, sqrtdt, nvar);
    //     almost<<<NB, NTPB, shared>>>(rho, cusum, gpu_kappa, gpu_theta, gpu_sigma, sqrtdt, nvar);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     float elapsed= 0;
    //     cudaEventElapsedTime(&elapsed, start, stop);
        
    //     float sum;
    //     cudaMemcpy(&sum, cusum, sizeof(float), cudaMemcpyDeviceToHost);
    //     sum /= tot;

    //     printf("for a value rho of %f we get a result E[(S-1)+] %f in %.3f ms\n", rho, sum, elapsed);
    //     cudaFree(cusum);
     
    // }
    // cudaFree(gpu_kappa);
    // cudaFree(gpu_theta);
    // cudaFree(gpu_sigma);



//#################uncomment for ex3###########################

    std::vector<float> kappas;
    std::vector<float> sigmas;
    std::vector<float> thetas;
    for (float kappa= 0.1; kappa<10; kappa += 0.5){
        for (float sigma= 0.1; sigma< 1; sigma += 0.05){
            for (float theta= 0.01; theta< 0.5; theta += 0.025){
                if (20 * kappa * theta > sigma * sigma){
                    kappas.push_back(kappa);
                    sigmas.push_back(sigma);
                    thetas.push_back(theta);
                }
            }
        }
    }


    int N_size= kappas.size();

    float *gpu_kappa, *gpu_theta, *gpu_sigma;

    cudaMalloc(&gpu_kappa, N_size * sizeof(float));
    cudaMalloc(&gpu_sigma, N_size * sizeof(float));
    cudaMalloc(&gpu_theta, N_size * sizeof(float));

    cudaMemcpy(gpu_kappa, kappas.data(),
           N_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_sigma, sigmas.data(),
            N_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_theta, thetas.data(),
            N_size * sizeof(float), cudaMemcpyHostToDevice);

    int NTPB= 512;
    int NB= N_size;
    size_t shared= NTPB * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::ofstream out("results.csv");
    out << "kappa,theta,sigma,rho,NTPB,time,euler,almost_exact,dt,result\n";
    
    printf("computing begins\n");

    for (float rho= -0.9; rho< 0.9; rho += 0.2){
        float nvar= sqrtf(1 - rho*rho);
        float *result;
        cudaMalloc(&result, N_size * sizeof(float));
        cudaMemset(result, 0, N_size * sizeof(float));

        cudaEventRecord(start);
        almost<<<NB, NTPB, shared>>>(rho, result, gpu_kappa, gpu_theta, gpu_sigma, sqrtdt, nvar);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float almost_elapsed= 0;
        cudaEventElapsedTime(&almost_elapsed, start, stop);
        cudaDeviceSynchronize();

        float* resCPU= (float*)malloc(N_size * sizeof(float));
        cudaMemcpy(resCPU, result, N_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(result);


        for (int i = 0; i < N_size; i++){
            out << kappas[i] << ","
                << thetas[i] << ","
                << sigmas[i] << ","
                << rho << ","
                << NTPB << ","
                << almost_elapsed << ","
                << 0 << ","
                << 1 << ","
                << dt << ","
                << resCPU[i] << ",\n";
        }


        float *resultE;
        cudaMalloc(&resultE, N_size * sizeof(float));
        cudaMemset(resultE, 0, N_size * sizeof(float));

        cudaEventRecord(start);
        Euler<<<NB, NTPB, shared>>>(rho, resultE, gpu_kappa, gpu_theta, gpu_sigma, sqrtdt, nvar);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float euler_elapsed= 0;
        cudaEventElapsedTime(&euler_elapsed, start, stop);
        cudaDeviceSynchronize();

        float* eulerCPU= (float*)malloc(N_size * sizeof(float));
        cudaMemcpy(eulerCPU, resultE, N_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(resultE);
        for (int i = 0; i < N_size; i++){
            out << kappas[i] << ","
                << thetas[i] << ","
                << sigmas[i] << ","
                << rho << ","
                << NTPB << ","
                << euler_elapsed << ","
                << 1 << ","
                << 0 << ","
                << dt << ","
                << eulerCPU[i] << ",\n";
        }
        free(eulerCPU);
        printf("results found in time %f ms for euler and %f ms for the almost exact methode\n", euler_elapsed, almost_elapsed);
        
    }
    out.close();

    cudaFree(gpu_kappa);
    cudaFree(gpu_sigma);
    cudaFree(gpu_theta);



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

