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


#include "wb-solve.h"

namespace wb
{
	bool Solve::Run(const double *pInitParams /* = NULL */)
	{
		if (!m_pfunc) {
			lout_Solve << "m_pFunc == NULL" << endl;
			return false;
		}
		
		Clock ck; 
		SAFE_DELETE_ARRAY(m_pdRoot);
		//lout_variable(m_pfunc->GetParamNum());
		m_pdRoot = new double[m_pfunc->GetParamNum()];

		double *pdCurParams = new double[m_pfunc->m_nParamNum]; //当前参数x_k
		double *pdCurGradient = new double[m_pfunc->m_nParamNum]; //当前的梯度 df_k
		double dCurValue = 0; // 函数值 f_k
		double *pdDir = new double[m_pfunc->m_nParamNum]; //方向,p_k
		double nExValueNum; // 额外数据的大小
		double dExValues[Func::cn_exvalue_max_num]; // 保存额外数据

		double dStep = 0; ///< 保存迭代步长
		double dGradNorm = 0; //梯度的模

		m_dSpendMinute = 0;

		//初始化
		for (int i = 0; i < m_pfunc->m_nParamNum; i++) {
			pdCurParams[i] = (pInitParams) ? pInitParams[i] : 1;
		}
		memset(pdDir, 0, sizeof(double)*m_pfunc->m_nParamNum);
		memset(pdCurGradient, 0, sizeof(double)*m_pfunc->m_nParamNum);

		IterInit(); ///init
		
		// iteration begin
		for (m_nIterNum = m_nIterMin; m_nIterNum <= m_nIterMax; m_nIterNum++)
		{
			lout_Solve << "======== iter:" << m_nIterNum << " ===(" << m_dSpendMinute << "m)=======" << endl;
			ck.Begin();

			// set the parameter
			m_pfunc->SetParam(pdCurParams);

			// get the gradient
			m_pfunc->GetGradient(pdCurGradient);
			// get the function value
			dCurValue = m_pfunc->GetValue();
			// get the ex-value
			nExValueNum = m_pfunc->GetExtraValues(m_nIterNum, dExValues);
			
			/* output the values */
			{
				lout_Solve << "dir_k={ ";
				for (int i = 0; i < min(4, m_pfunc->m_nParamNum); i++)
					lout << pdDir[i] << " ";
				lout << "... }" << endl;

				lout_Solve << "x_k={ ";
				for (int i = 0; i < min(4, m_pfunc->m_nParamNum); i++)
					lout << pdCurParams[i] << " ";
				lout << "... }" << endl;

				lout_Solve << "g_k={ ";
				for (int i = 0; i < min(4, m_pfunc->m_nParamNum); i++)
					lout << pdCurGradient[i] << " ";
				lout << "... }" << endl;

				double dNorm = VecNorm(pdCurGradient, m_pfunc->m_nParamNum);
				lout_Solve << "a=" << dStep << " |g_k|=" << dNorm << " f_k=" << dCurValue << endl;
				
				lout_Solve << "ExValues={ ";
				for (int i = 0; i < nExValueNum; i++)
					lout << dExValues[i] << " ";
				lout << "}" << endl;

// #ifdef _DEBUG
// 				Pause();
// #endif
			}

			/* Stop Decision */
			if (StopDecision(m_nIterNum, dCurValue, pdCurGradient)) {
				break;
			}

			
			// get the update direction
			ComputeDir(m_nIterNum, pdCurParams, pdCurGradient, pdDir);

			// Line search
			if (m_dGain > 0)
				dStep = m_dGain;
			else
				dStep = LineSearch(pdDir, dCurValue, pdCurParams, pdCurGradient);

			// Update parameters
			Update(pdCurParams, pdDir, dStep);

			// Add the spend times
			m_dSpendMinute += ck.ToSecond(ck.End()) / 60;
		}


		lout_Solve << "======== iter:" << m_nIterNum << " ===(" << m_dSpendMinute << "m)=======" << endl;
		lout_Solve << "Iter Finished!" << endl;

		// Save the result
		memcpy(m_pdRoot, pdCurParams, sizeof(m_pdRoot[0])*m_pfunc->m_nParamNum);


		SAFE_DELETE_ARRAY(pdDir);
		SAFE_DELETE_ARRAY(pdCurGradient);
		SAFE_DELETE_ARRAY(pdCurParams);

		return true;
	}
	void Solve::ComputeDir(int k, const double *pdParam, const double *pdGradient, double *pdDir)
	{
		/* gradient descent */
		for (int i = 0; i < m_pfunc->m_nParamNum; i++)
			pdDir[i] = -pdGradient[i];
	}
	double Solve::LineSearch(double *pdDir, double dValue, const double *pdCurParam, const double *pdGradient)
	{
		/*
		需要额外的SetParam的代价
		*/
		double *pdNextParam = new double[m_pfunc->m_nParamNum];

		/// 算法参加Numerical Optimization，P57，介绍的interpolation算法
		double a0 = 0, a1 = 0, a2 = 0; //步长
		double phi0 = 0, phi1 = 0, phi2 = 0; //phi(ai)
		double c = 1e-4;
		double phi_t = VecProduct(pdGradient, pdDir, m_pfunc->m_nParamNum); // phi'(0)

		a2 = 1.0; //初始步长设为1
		for (int k = 1; a2 > 0; k++)
		{
			// x = x + a * p
			for (int n = 0; n < m_pfunc->m_nParamNum; n++)
				pdNextParam[n] = pdCurParam[n] + a2 * pdDir[n];

			m_pfunc->SetParam(pdNextParam);
			phi2 = m_pfunc->GetValue(); // phi(a2)

			if (phi2 <= dValue + c * a2 * phi_t)
				break;

			//保存前两次的结果
			a0 = a1;
			a1 = a2;
			phi0 = phi1;
			phi1 = phi2;
			if (k == 1) {
				a2 = -phi_t*a1*a1 / 2 / (phi1 - dValue - phi_t * a1);
			}
			else {
				double v1 = phi1 - dValue - phi_t * a1;
				double v2 = phi0 - dValue - phi_t * a0;
				double a = a0*a0*v1 - a1*a1*v2;
				double b = -a0*a0*a0*v1 + a1*a1*a1*v2;
				double t = a0*a0*a1*a1*(a1 - a0);
				a /= t;
				b /= t;
				a2 = (-b + sqrt(b*b - 3 * a*phi_t)) / (3 * a);
			}

			//如果a2与a1太接近或差的太远，则取a2=a1/2
			if (fabs(a2 - a1) < 1e-5 ||
				a1 / a2 > 10)
				a2 = a1 / 2;
			//如果a2过小，则取一个不太小的值
			if (a2 < 1e-10) {
				lout_warning("[Solve] LineSearch: a2 is too small < 1e-10, break")
					a2 = 1e-5;
				break;
			}
		}

		SAFE_DELETE_ARRAY(pdNextParam);

		return a2;
	}
	void Solve::Update(double *pdParam, const double *pdDir, double dStep)
	{
		for (int i = 0; i < m_pfunc->m_nParamNum; i++) {
			pdParam[i] += pdDir[i] * dStep;
		}
	}
	bool Solve::StopDecision(int k, double dValue, const double *pdGradient)
	{
		if (VecNorm(pdGradient, m_pfunc->m_nParamNum) < m_dStop) {
			return true;
		}
		if (k == m_nIterMax) { //防止因为迭代次数终止后，又进行一次额外的迭代
			return true;
		}
		return false;
	}
	double Solve::VecProduct(const double *pdVec1, const double *pdVec2, int nSize)
	{
		double d = 0;
		for (int i = 0; i < nSize; i++)
			d += pdVec1[i] * pdVec2[i];
		return d;
	}
	double Solve::VecNorm(const double *pdVec, int nSize)
	{
		return sqrt(VecProduct(pdVec, pdVec, nSize));
	}
	double Solve::VecDist(const double *pdVec1, const double *pdVec2, int nSize)
	{
		double d = 0;
		for (int i = 0; i < nSize; i++)
			d += pow(pdVec1[i] - pdVec2[i], 2);
		return sqrt(d);
	}



	void LBFGS::IterInit()
	{
		SAFE_DELETE_ARRAY(m_pdPrevParam);
		SAFE_DELETE_ARRAY(m_pdPrevGradient);
		SAFE_DELETE_ARRAY(m_pdAlpha);
		
		m_pdPrevParam = new double[m_pfunc->GetParamNum()];
		m_pdPrevGradient = new double[m_pfunc->GetParamNum()];
		m_pdAlpha = new double[m_nLimitiedNum];

		CirQueueBuf_Release();
		CirQueueBuf_Init();
	}
	void LBFGS::ComputeDir(int k, const double *pdParam, const double *pdGradient, double *pdDir)
	{
		if (k > 1) {
			// 保存用于LBFGS的vector
			CirQueueBuf_In(m_pd_s, m_pd_y);
			for (int n = 0; n < m_pfunc->m_nParamNum; n++) {
				m_pd_s[n] = pdParam[n] - m_pdPrevParam[n];
				m_pd_y[n] = pdGradient[n] - m_pdPrevGradient[n];
			}
		}

		/*
		计算LBFGS direction
		*/

		double *pd_s = NULL;
		double *pd_y = NULL;
		int nVecLen = m_pfunc->m_nParamNum;

		lout_Solve << "LBFGS dir computer" << endl;
		int nBound = min(m_nLimitiedNum, k-1); //最多只计算前m个结果

		//将梯度赋给q
		memcpy(pdDir, pdGradient, sizeof(pdDir[0])*nVecLen);


		//确保新来的向量不能为0向量
		if (nBound >= 1) {
			CirQueueBuf_Prev(1, pd_s, pd_y);
		}



		//第一个循环
		for (int i = 1; i <= nBound; i++)
		{
			CirQueueBuf_Prev(i, pd_s, pd_y);

			double dProd = VecProduct(pd_s, pd_y, nVecLen);
			m_pdAlpha[i - 1] = VecProduct(pd_s, pdDir, nVecLen) / dProd;
			for (int n = 0; n < nVecLen; n++)
				pdDir[n] -= m_pdAlpha[i - 1] * pd_y[n];
		}


		//计算gamma,即初始的H^0
		double dGamma = 1;
		if (k > 1) {
			CirQueueBuf_Prev(1, pd_s, pd_y);
			dGamma = VecProduct(pd_s, pd_y, nVecLen) / VecProduct(pd_y, pd_y, nVecLen);
		}

		// r = H^0 * q
		for (int n = 0; n < nVecLen; n++)
			pdDir[n] *= dGamma;

		//第二个循环
		for (int i = nBound; i >= 1; i--)
		{
			CirQueueBuf_Prev(i, pd_s, pd_y);
			double dBeta = VecProduct(pd_y, pdDir, nVecLen) / VecProduct(pd_y, pd_s, nVecLen);
			for (int n = 0; n < nVecLen; n++)
				pdDir[n] += pd_s[n] * (m_pdAlpha[i - 1] - dBeta);
		}



		//方向需要取反
		for (int n = 0; n < nVecLen; n++)
			pdDir[n] = -pdDir[n];



		/*
		Save the previsou parameter and gradient
		*/
		memcpy(m_pdPrevParam, pdParam, sizeof(pdParam[0])*m_pfunc->m_nParamNum);
		memcpy(m_pdPrevGradient, pdGradient, sizeof(pdGradient[0])*m_pfunc->m_nParamNum);
	}

	void LBFGS::CirQueueBuf_Init()
	{
		m_pCirQueueBuf = new sy[m_nLimitiedNum];
		for (int i = 0; i < m_nLimitiedNum; i++) {
			m_pCirQueueBuf[i].s = new double[m_pfunc->m_nParamNum];
			m_pCirQueueBuf[i].y = new double[m_pfunc->m_nParamNum];
		}
		m_nCirQueueBufTail = 0;
	}
	void LBFGS::CirQueueBuf_Release()
	{
		if (m_pCirQueueBuf) {
			for (int i = 0; i < m_nLimitiedNum; i++) {
				SAFE_DELETE_ARRAY(m_pCirQueueBuf[i].s);
				SAFE_DELETE_ARRAY(m_pCirQueueBuf[i].y);
			}
			SAFE_DELETE_ARRAY(m_pCirQueueBuf);
		}
		m_nCirQueueBufTail = 0;
	}
	void LBFGS::CirQueueBuf_Prev(int i, double *&pd_s, double *&pd_y)
	{
		i = (m_nLimitiedNum + m_nCirQueueBufTail - i) % m_nLimitiedNum;
		pd_s = m_pCirQueueBuf[i].s;
		pd_y = m_pCirQueueBuf[i].y;
	}
	void LBFGS::CirQueueBuf_In(double *&pd_s, double *&pd_y)
	{
		pd_s = m_pCirQueueBuf[m_nCirQueueBufTail].s;
		pd_y = m_pCirQueueBuf[m_nCirQueueBufTail].y;
		m_nCirQueueBufTail = (m_nCirQueueBufTail + 1) % m_nLimitiedNum;
	}
}