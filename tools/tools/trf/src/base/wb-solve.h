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

/**
* \file
* \author WangBin
* \date 2016-04-28
* \brief define the framework of iterative algorithms, such as gradient descent or LBFGS.
*/

#pragma once
#include "wb-win.h"
#include "wb-log.h"
#include "wb-vector.h"
#include <algorithm>

namespace wb
{
	class Solve;
	/**
	* \class Func
	* \author wangbin
	* \date 2016-03-04
	* \brief the objective function, used to derive
	*/
	class Func
	{
		friend class Solve;
		friend class LBFGS;
	protected:
		Solve *m_pSolve; ///< Save the solve pointor
		int m_nParamNum; ///< the parameter number
	public:
		
		Func(int nParamNum = 0):m_pSolve(NULL) { SetParamNum(nParamNum); };
		/// setting the parameter number
		void SetParamNum(int n) { m_nParamNum = n; }
		/// get the paremeter number
		int GetParamNum() const { return m_nParamNum; }
		/// set the parameter.
		virtual void SetParam(double *pdParams) = 0;
		/// calculate the function value f(x)
		virtual double GetValue() = 0;
		/// calculate the gradient g(x)
		virtual void GetGradient(double *pdGradient) = 0;
		static const int cn_exvalue_max_num = 100; ///< the maximu number of the values returned by GetExtrValues
		/// calculate extra values which will be print at each iteration
		/**
		* \param[in] k iteration number form 1 to ...
		* \param[out] pdValues	Return the values needed to be outputed. The memory is allocated outside and the maximum size = cn_exvalue_max_num
		* \return return the pdValues number
		*/
		virtual int GetExtraValues(int k, double *pdValues) { return 0; }
	};

#define lout_Solve wb::lout<<m_pAlgorithmName<<" "

	/**
	 * \class Solve
	 * \author wangbin
	 * \date 2016-03-04
	 * \brief the base class of all the solve classes, and provide a gradient descent algorithm.
	 */
	class Solve
	{
	protected:
		const char *m_pAlgorithmName; ///< the algorithm name.
	public:
		Func *m_pfunc; ///< pointer to the function

		double *m_pdRoot; ///< save the root of the function

		int m_nIterNum; ///< current iteration number, iter form m_nIterMin to m_nIterMax
		int m_nIterMin; ///< minium iteration number
		int m_nIterMax; ///< maximum iteration number

		double m_dSpendMinute; ///< record the iteration spend time��minute��

		double m_dStop; ///< stop threshold
		double m_dGain; ///< itera step. ==0 means using the line search .

	public:
		Solve(Func *pfunc = NULL, double dtol=1e-5) 
		{ 
			m_pAlgorithmName = "[Solve]";
			m_pfunc = pfunc;
			m_pfunc->m_pSolve = this;
			m_pdRoot = NULL;

			m_nIterNum = 0;
			m_nIterMin = 1;
			m_nIterMax = 10000;

			m_dStop = dtol;
			m_dGain = 0;
		}
		
		/// Run iteration. input the init-parameters.
		virtual bool Run(const double *pInitParams = NULL);
		/// initial the iteration, for derivation.
		virtual void IterInit() {};
		/// Calculate the update direction p_k
		/**
		* \param [in] k iteration number, from 1 to ...
		* \param [in] pdParam parameter x_k
		* \param [in] pdGradient the gradient at x_k
		* \param [out] pdDir return the direction
		*/
		virtual void ComputeDir(int k, const double *pdParam, const double *pdGradient, double *pdDir);
		/// linear search.
		/**
		* \param [in] pdDir update direction p_k
		* \param [in] dValue the function value at x_k
		* \param [in] pdGradient the gradient at x_k
		* \param [in] pdCurParam parameter x_k
		* \return  the learning step.
		*/
		virtual double LineSearch(double *pdDir, double dValue, const double *pdParam, const double *pdGradient);
		/// Update 
		virtual void Update(double *pdParam, const double *pdDir, double dStep);
		/// Stop decision
		virtual bool StopDecision(int k, double dValue, const double *pdGradient);

	public:
		/// calculate the dot of two vectors
		static double VecProduct(const double *pdVec1, const double *pdVec2, int nSize);
		/// calculate the norm of a vector
		static double VecNorm(const double *pdVec, int nSize);
		/// calculate the distance of two vectors
		static double VecDist(const double *pdVec1, const double *pdVec2, int nSize);
	};

	/*
	 * \class LBFGS
	 * \brief LBFGS method
	*/
	class LBFGS : public Solve
	{
		/// store the delta_x and delta_gradient.
		typedef struct {
			double *s;
			double *y;
		} sy;
	protected:
		int m_nLimitiedNum; ///< limited number, i.e. m
		sy *m_pCirQueueBuf; ///< the buffer of circular queue to store s_k = x_k - x_(k-1) and y_k = g_k - g_{k-1}
		int m_nCirQueueBufTail; ///< queue tail
		double *m_pd_s, *m_pd_y; ///< current s_k = x_k - x_{k-1} and y_k 

		double *m_pdPrevGradient; ///< gradient on previous iteration
		double *m_pdPrevParam;	  ///< parameter on previous iteration
		double *m_pdAlpha;        ///< auxillary factor in ComputeDir

	public:
		/// constructor
		LBFGS(Func *pfunc = NULL, double dtol = 1e-5) :Solve(pfunc, dtol)
		{
			m_nLimitiedNum = 8;
			m_pCirQueueBuf = NULL;
			m_nCirQueueBufTail = 0;
			m_pd_s = NULL;
			m_pd_y = NULL;

			m_pdPrevGradient = NULL;
			m_pdPrevParam = NULL;
			m_pdAlpha = NULL;
		}
		/// destructor
		~LBFGS()
		{
			SAFE_DELETE_ARRAY(m_pdPrevParam);
			SAFE_DELETE_ARRAY(m_pdPrevGradient);
			SAFE_DELETE_ARRAY(m_pdAlpha);

			CirQueueBuf_Release();
		}
		/// iter init
		virtual void IterInit();
		/// Calculate the update direction p_k, referring to "Numerical Optimization"��P178��Algorithm 7.4.
		virtual void ComputeDir(int k, const double *pdParam, const double *pdGradient, double *pdDir);
		/// Init the circular queue
		void CirQueueBuf_Init();
		/// release the circular queue
		void CirQueueBuf_Release();
		/// find the previoud ith datas, i<=m_nLimitiedNum
		void CirQueueBuf_Prev(int i, double *&pd_s, double *&pd_y);
		/// in queue. return the pointer.
		void CirQueueBuf_In(double *&pd_s, double *&pd_y);
	};
}
