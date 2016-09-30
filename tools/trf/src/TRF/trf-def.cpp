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
// All h, cpp, cc, and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.


#include "trf-def.h"

namespace trf
{

	int omp_rand(int thread_num)
	{
#ifdef __linux
		static unsigned int s_states[128];
		if (thread_num == -1) {

			return rand_r(&s_states[omp_get_thread_num()]);
		}
		else {
			if (thread_num > 128) {
				lout_error("[Rand] Can not support to many thread (Over 128)");
			}
			lout << "[Rand] Initial State" << endl;
			srand(time(NULL));
			for (int i = 0; i < thread_num; i++) {
				s_states[i] = rand();
			}
		}
#else
		if (thread_num != -1) {
			srand(time(NULL));
		}
		return rand();
#endif
	}

	/************************************************************************/
	/* The definition of sampling function                                  */
	/************************************************************************/

	LogP LogLineNormalize(LogP* pdProbs, int nNum)
	{
		LogP dSum = LogP_zero;
		for (int i = 0; i < nNum; i++)
			dSum = Log_Sum(dSum, pdProbs[i]);
		for (int i = 0; i < nNum; i++)
			pdProbs[i] -= dSum;
		return dSum;
	}
	int LogLineSampling(const LogP* pdProbs, int nNum)
	{
		double d = 1.0 * omp_rand() / RAND_MAX;
		int sX = 0;
		double dSum = 0;

		for (sX = 0; sX<nNum; sX++) {
			dSum += LogP2Prob(pdProbs[sX]);

			if (fabs(dSum - 1) < 1e-5)
				dSum = 1; //确保精度
			if (dSum == 0)
				continue; //0概率

			if (d <= dSum)
				break;
		}
		if (sX >= nNum) {
// 			cout << "[LogLineSampling] " << sX << "\t" << dSum << endl;
// 			lout_assert(sX<nNum);
			sX = nNum - 1;
		}

		return sX;
	}
	void LineNormalize(Prob* pdProbs, int nNum)
	{
		Prob dSum = 0;
		for (int i = 0; i<nNum; i++)
			dSum += pdProbs[i];

		if (dSum > 0) {
			for (int i = 0; i<nNum; i++)
				pdProbs[i] /= dSum;
		}
		else {
			for (int i = 0; i<nNum; i++)
				pdProbs[i] = 1.0 / nNum;
		}

	}
	int LineSampling(const Prob* pdProbs, int nNum)
	{
		double d = 1.0 * omp_rand() / RAND_MAX;
		int sX = 0;
		double dSum = 0;

		for (sX = 0; sX<nNum; sX++) {
			dSum += pdProbs[sX];

			if (fabs(dSum - 1) < 1e-5)
				dSum = 1; //确保精度
			if (dSum == 0)
				continue; //0概率

			if (d <= dSum)
				break;
		}
		if (sX >= nNum) {
			cout << "[LineSampling] " << sX << "\t" << dSum << endl;
			lout_assert(sX<nNum);
		}

		return sX;
	}
	bool Acceptable(Prob prob)
	{
		double d = 1.0 * omp_rand() / RAND_MAX;
		return d <= prob;
	}
	void RandomPos(int *a, int len, int n)
	{
		if (n> len) {
			lout_error("[RandomPos] n(" << n << ") > len(" << len << ") !!");
		}

		for (int i = 0; i<n; i++) {
			int s = rand() % (len - i);
			//将a中第i+s个位置与第i位置交换
			int temp = a[i + s];
			a[i + s] = a[i];
			a[i] = temp;
		}
	}

	double dRand(double dmin, double dmax)
	{
		double d = 1.0 * rand() / RAND_MAX;
		return  d*(dmax - dmin) + dmin;
	}
	int omp_nrand(int nMin, int nMax)
	{
		return omp_rand() % (nMax - nMin) + nMin;
	}

	void EasySmooth(Prob *p, int num, Prob threshold /*= 1e-5*/)
	{
		for (int i = 0; i < num; i++) {
			p[i] = max(p[i], threshold);
		}
		LineNormalize(p, num);
	}

	double GetAISFactor(int t, int T)
	{
		t = t + 1; /* make sure that t is from 1 to T */

		double delta = 4;
		double b_t = SigmFunc( delta * (2.0 * t / T - 1) );
		double b_T = SigmFunc( delta * (2.0 * T / T - 1) );
		double b_1 = SigmFunc( delta * (2.0 * 1 / T - 1) );

		return (b_t - b_1) / (b_T - b_1);
	}

	double SigmFunc(double x)
	{
		return 1.0 / (1 + exp(-x));
	}

}