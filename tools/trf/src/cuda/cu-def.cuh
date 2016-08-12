// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2014-2015 Tsinghua University
// Author: wb.th08@gmail.com (Bin Wang), ozj@tsinghua.edu.cn (Zhijian Ou) 
//
// All h, cpp, cc, cu, cuh and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.

/**
* \file
* \author WangBin
* \date 2016-05-19
* \brief some definition for cude codes.
*/

#ifndef _CU_DEF_CUH_
#define _CU_DEF_CUH_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <iostream>
#include "wb-vector.h"
using namespace std;

#define CUDA_CALL(x) \
	if (cudaSuccess != (x)) {\
		printf("[CudaError] call="#x"\n"); \
		printf("[CudaError] error = %s\n",  cudaGetErrorName(cudaGetLastError())); \
		exit(1); \
	}

#define CURAND_CALL(x) \
	if((x)!=CURAND_STATUS_SUCCESS) {\
		printf("[CudaError] call="#x"\n"); \
		exit(1); \
	}


	

#define CUDA_SAFE_FREE(p) \
	if (p) {\
		CUDA_CALL(cudaFree(p)) \
		p = NULL; \
	}

namespace cu
{
	typedef enum {none, host, device} MemLocation;

	struct Reg
	{
		int nBeg;
		int nEnd;
	};

	/*
	 * \class Array
	 * \brief Record the number, but the buffer of Array should be allocated beforehand
	*/
	template <typename T>
	class Array
	{
	private:
		T *m_pBuf; ///< buffer pointer
		int m_nTop; ///< the top of current array
		int m_nBuf; ///< the buffer size
	public:
		__host__ __device__ Array(int bufsize = 0) :m_pBuf(NULL), m_nTop(-1), m_nBuf(0) { Reset(bufsize); }
		__host__ __device__ Array(T* pbuf, int nBuf, int nTop = -1) :m_pBuf(pbuf), m_nBuf(nBuf), m_nTop(nTop) {}
		__host__ Array(wb::Array<T> &a) :m_pBuf(NULL), m_nTop(-1), m_nBuf(0) { Copy(a); }
		__host__ void Reset(int bufsize = 0) {
			CUDA_SAFE_FREE(m_pBuf);
			m_nTop = -1;
			m_nBuf = 0;

			if (bufsize > 0) {
				CUDA_CALL(cudaMalloc(&m_pBuf, sizeof(T)*bufsize));
				m_nBuf = bufsize;
			}
		}
		__host__ void Copy(wb::Array<T> &a) {
			Reset(a.GetNum());
			m_nTop = a.GetNum() - 1;
			CUDA_CALL(cudaMemcpy(m_pBuf, a.GetBuffer(), sizeof(T)*a.GetNum(), cudaMemcpyHostToDevice));
		}
		__host__ void CopyTo(wb::Array<T> &a) {
			a.SetNum(GetNum());
			CUDA_CALL(cudaMemcpy(a.GetBuffer(), m_pBuf, sizeof(T)*GetNum(), cudaMemcpyDeviceToHost));
		}
		__device__ T& operator [] (int i) { 
//#ifdef _DEBUG
			if (i >= m_nBuf) {
				printf( "ERROR [Array] op[]: Over Flow (i=%d, m_nBuf=%d)\n", i, m_nBuf);
				return m_pBuf[0];
			}
//#endif
			m_nTop = max(m_nTop, i); 
			return m_pBuf[i];
		}
		__device__ void Add(T t) {
//#ifdef _DEBUG
			if (m_nTop >= m_nBuf) {
				printf("ERROR [Array] Add: Over Flow (m_nTop=%d, m_nBuf=%d)\n", m_nTop, m_nBuf);
				return;
			}
//#endif
			m_nTop++;
			m_pBuf[m_nTop] = t;
		}
		__host__ __device__ void Clean() { m_nTop = -1; }
		__host__ __device__ int GetNum() const { return m_nTop + 1;}
		__host__ __device__ T* GetBuf() const { return m_pBuf; }
	};

	/*
		Rules for Cuda class defination:
		- Never free the memory in the destuctor function, (~LHash())
		- Define a function called Copy() to copy a corresponding class to device memory.
	*/

	/*
	 * \brief memory pool, used to allocate memory from a large memory pool.
	 * \detials
		The global buffer in Pool should be allocated or freed manually, by calling functions BufMalloc or BufFree;\n
		The global buffer can also be maintained outside the class.
	*/
	class Pool
	{
	private:
		char *m_pbuf;  ///< pointer to buffer
		size_t m_size; ///< the buffer total size
		size_t m_cur;  ///< the current pointer to rest of access buffer
	public:
		__host__ __device__ Pool() : m_pbuf(NULL), m_size(0), m_cur(0) {}
		__host__ __device__ Pool(char *pbuf, size_t size) : m_pbuf(pbuf), m_size(size), m_cur(0) {}
		__host__ __device__ ~Pool() {};
		__host__ char* New(size_t needsize)
		{
			if (m_cur + needsize > m_size) {
				cerr << "[ERROR] [cuPool] New: there are no enough memory";
				cerr << "(size = " << m_size << " cur = " << m_cur << " need=" << needsize << endl;
				return NULL;
			}
			char *p = m_pbuf + m_cur;
			m_cur += needsize;
			return p;
		}
		__host__ void BufMalloc(int nsize)
		{
			CUDA_CALL(cudaMalloc(&m_pbuf, nsize));
		}
		__host__ void BufFree()
		{
			CUDA_SAFE_FREE(m_pbuf);
		}
	};
}

#endif