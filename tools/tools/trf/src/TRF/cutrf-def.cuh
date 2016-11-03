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

#ifndef _CUTRF_DEF_CUH_
#define _CUTRF_DEF_CUH_
#include "cu-def.cuh"
#include "cu-lhash.cuh"
#include "cu-mat.cuh"
#include "cu-trie.cuh"
#include "cu-string.cuh"
#include "trf-def.h"

namespace cutrf
{
	typedef trf::Prob Prob;
	typedef trf::LogP LogP;

	typedef curandState RandState;

#define LOGP_ZERO -1e20

#define _TID (blockIdx.x * blockDim.x + threadIdx.x)


	__host__ __device__ Prob LogP2Prob(LogP x) {
		return (x <= LOGP_ZERO / 2) ? 0 : exp((double)(x));
	}
	__host__ __device__ LogP Prob2LogP(Prob x) {
		return ((x) <= 0) ? LOGP_ZERO : log((double)(x));
	}
	__host__ __device__ LogP Log_Sum(LogP x, LogP y) {
		return (x > y) ? x + Prob2LogP(1 + LogP2Prob(y - x)) : y + Prob2LogP(1 + LogP2Prob(x - y));
	}
	__host__ __device__ LogP Log_Sub(LogP x, LogP y) {
		return (x > y) ? x + Prob2LogP(1 - LogP2Prob(y - x)) : y + Prob2LogP(LogP2Prob(x - y) - 1);
	}
	/// log summate all the values in array
	__host__ __device__ LogP Log_Sum(LogP *p, int num) {
		LogP sum = LOGP_ZERO;
		for (int i = 0; i < num; i++) {
			sum = Log_Sum(sum, p[i]);
		}
		return sum;
	}

	__device__ LogP LogLineNormalize(LogP* pdProbs, int nNum)
	{
		LogP dSum = LOGP_ZERO;
		for (int i = 0; i < nNum; i++)
			dSum = Log_Sum(dSum, pdProbs[i]);
		for (int i = 0; i < nNum; i++)
			pdProbs[i] -= dSum;
		return dSum;
	}
	__device__ int LogLineSampling(const LogP* pdProbs, int nNum, RandState *state)
	{
		float d = curand_uniform(state);
		int sX = 0;
		float dSum = 0;

		for (sX = 0; sX < nNum; sX++) {
			dSum += LogP2Prob(pdProbs[sX]);

			if (fabs(dSum - 1) < 1e-5)
				dSum = 1; //确保精度
			if (dSum == 0)
				continue; //0概率

			if (d <= dSum)
				break;
		}
		if (sX >= nNum) {
			printf("[ERROR] [LogLineSampling] sX(%d) >= nNum(%d)\n", sX, nNum);
		}

		return sX;
	}
	__device__ void LineNormalize(Prob* pdProbs, int nNum)
	{
		Prob dSum = 0;
		for (int i = 0; i < nNum; i++)
			dSum += pdProbs[i];

		if (dSum > 0) {
			for (int i = 0; i < nNum; i++)
				pdProbs[i] /= dSum;
		}
		else {
			for (int i = 0; i < nNum; i++)
				pdProbs[i] = 1.0 / nNum;
		}

	}
	__device__ int LineSampling(const Prob* pdProbs, int nNum, RandState *state)
	{
		float d = curand_uniform(state);
		int sX = 0;
		float dSum = 0;

		for (sX = 0; sX < nNum; sX++) {
			dSum += pdProbs[sX];

			if (fabs(dSum - 1) < 1e-5)
				dSum = 1; //确保精度
			if (dSum == 0)
				continue; //0概率

			if (d <= dSum)
				break;
		}
		if (sX >= nNum) {
			printf("[ERROR] [LineSampling] sX(%d) >= nNum(%d)\n", sX, nNum);
		}

		return sX;
	}
	__device__ bool Acceptable(Prob prob, RandState *state)
	{
		float d = curand_uniform(state);
		return d <= prob;
	}

	__global__ void Kernal_InitState(RandState *state, int n, unsigned long long seed)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			curand_init(seed, tid, 0, &state[tid]);
		}
	}
}

#endif