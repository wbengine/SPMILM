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


#pragma once
#include "trf-ml-train.h"
#include <omp.h>


namespace trf
{
	class SAfunc;
	class SAtrain;

	/* the AIS configurations */
	class AISConfig
	{
	public:
		int nChain; ///< chain number
		int nInter; ///< intermediate distribution number
		AISConfig(int p_nChain = 16, int p_nInter = 1000) :nChain(p_nChain), nInter(p_nInter){}
		void Parse(const char* str) {
			String tempStr(str);
			char *p = strtok(tempStr.GetBuffer(), ":,");
			nChain = atoi(p);
			p = strtok(NULL, ":,");
			nInter = atoi(p);
		}
	};

	/* Save the last sequence of each length in each thread */
	class ThreadData
	{
	public:
		Array<Seq*> aSeqs;
	public:
		~ThreadData();
		void Create(int maxlen, Model *pModel);
	};

	
	/*
	 * \class
	 * \brief augment SA training algorithms
	*/
	class SAfunc : public MLfunc
	{
		friend class SAtrain;
	protected:
		int m_nMiniBatchSample;  ///< mini-batch for samples
		
		Vec<Prob> m_samplePi; ///< the length distribution used for sample

		Vec<double> m_vAllSampleLenCount; ///< the count of each length in all samples
		Vec<double> m_vCurSampleLenCount; ///< the count of length in samples of current iteration
		int m_nTotalSample; ///< the total sample number

//		Vec<double> m_vEmpiricalExp; ///< the empirical expectation
//		Vec<double> m_vEmpiricalExp2; ///< the empirical expectation E[f^2]
		Vec<double> m_vSampleExp; ///< the sample expectation
		Vec<double> m_vSampleExp2; ///< the sample expectation^2
		Vec<double> m_vSampleLen; ///< the sample length expectation


		//Array<ThreadData*> m_threadData; ///< save the last sequence of each threads
		Array<Seq*> m_threadSeq; ///< save the last sequence of each threads

// 		Mat<double> m_matEmpiricalExp; ///< the empirical expectation of each thread
// 		Mat<double> m_matEmpiricalExp2; ///< empirical E[f^2] of each thread
		Mat<double> m_matSampleExp; ///< the sample expectation of each thread
		Mat<double> m_matSampleExp2; ///< the sample expectation^2 of each thread
		Mat<double> m_matSampleLen; ///< the length count of sample of each thread

		Vec<double> m_vEmpiricalVar; ///< empirical variance 
		
	public:
		double m_fRegL2; ///< l2 regularization
		double m_var_gap; ///< a varicance gap used in gradient sacling

		AISConfig m_AISConfigForZ; ///< the AIS configuration for normalization
		AISConfig m_AISConfigForP; ///< the AIS configuration for calculating the LL.
		int m_nCDSampleTimes; ///< the CD-n: the sample number.
		int m_nSASampleTimes; ///< the SA sample times
		
	public:
		File m_fdbg;  ///< output the sample pi/zete information
		File m_fparm; ///< output the parameters of each iteration
		File m_fgrad; ///< output the gradient of each iteration
		File m_fmean; ///< output the p[f] on training set
		File m_fvar;  ///< output the variance at each iteration
		File m_fexp;  ///< output the expectation of each iteartion
		File m_fsamp; ///< output all the samples
		File m_ftrain;///< output all the training sequences
		File m_ftrainLL; ///< output loglikelihood on training set
		File m_fvallidLL;///< output loglikelihood on valid set
		File m_ftestLL;  ///< output loglikelihood on test set


	public:
		SAfunc() :m_nMiniBatchSample(100) {
			m_var_gap = 1e-15;
			m_fRegL2 = 0;
		};
		SAfunc(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL, int nMinibatch = 100 )
		{
			m_var_gap = 1e-15;
			m_fRegL2 = 0;
			Reset(pModel, pTrain, pValid, pTest, nMinibatch);
		}
		~SAfunc()
		{
#ifndef _CD
			for (int i = 0; i < m_threadSeq.GetNum(); i++) {
				SAFE_DELETE(m_threadSeq[i]);
			}
#endif
		}
		/// reset 
		virtual void Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL, int nMinibatch = 100);
		/// print information
		void PrintInfo();
		/// get the ngram feature number
		int GetFeatNum() const { return m_pModel->GetParamNum(); }
		/// get the zeta parameter number
		int GetZetaNum() const { return m_pModel->GetMaxLen() + 1; }
		/// get a random sequence
		void RandSeq(Seq &seq, int nLen = -1);
		/// get the parameters
		void GetParam(double *pdParams);
		/// calculate the empirical expectation
		void GetEmpVar(CorpusBase *pCorpus, Vec<double> &vVar);

		/// calcualte the expectation of SA samples
		virtual void GetSampleExp(VecShell<double> &vExp, VecShell<double> &vExp2, VecShell<double> &vLen);

		/// do something at the end of the SA iteration
		void IterEnd(double *pFinalParams);
		/// Write Model
		void WriteModel(int nEpoch);

		virtual void SetParam(double *pdParams);
		virtual void GetGradient(double *pdGradient);
		virtual double GetValue() { return 0; }
		virtual int GetExtraValues(int t, double *pdValues);
	};

	/*
	 * \class
	 * \brief Learning rate
	*/
	class LearningRate
	{
	public:
		double beta;
		double tc;
		double t0;
	public:
		LearningRate() :beta(1), tc(0), t0(0) {}
		void Reset(const char *pstr, int p_t0);
		/// input the iteration number, get the learning rate
		double Get(int t);
	};

	/*
	 * \class
	 * \brief SAtraining
	*/
	class SAtrain : public Solve
	{
	protected:
		double m_gamma_lambda;
		double m_gamma_zeta;

	public:
		LearningRate m_gain_lambda;
		LearningRate m_gain_zeta;
		//double m_zeta_upgap; ///< the gap for zeta update

		bool m_bUpdate_lambda;
		bool m_bUpdate_zeta;
		double m_dir_gap; ///< control the dir values

		int m_nAvgBeg; ///< if >0, then calculate the average
		float m_fEpochNun; ///< the current epoch number - double
		int m_nPrintPerIter;  ///< output the LL per iteration, if ==-1, the disable
		wb::Array<int> m_aWriteAtIter; ///< output temp model at some iteration

#ifdef _Adam
		double adam_beta1;
		double adam_beta2;
		double adam_sigma;
		double adam_alpha;
		Vec<double> adam_m; ///< moving averaged gradient
		Vec<double> adam_v; ///< moving averaged gradient^2
#endif

#ifdef _Hession
		Vec<double> m_avgHes; ///< the moveing averaged hession
#endif

	public:
		SAtrain(SAfunc *pfunc = NULL) : Solve(pfunc)
		{
			m_pAlgorithmName = "[SAMS]";

			m_gamma_lambda = 1;
			m_gamma_zeta = 1;
			//m_zeta_upgap = 10;


			m_bUpdate_lambda = true;
			m_bUpdate_zeta = true;
			m_dir_gap = 1.0;

			m_nAvgBeg = 0;

			m_fEpochNun = 0;
			m_nPrintPerIter = 1;
#ifdef _Hession
			m_avgHes.Reset(pfunc->GetFeatNum());
			m_avgHes.Fill(0);
#endif
#ifdef _Adam
			adam_beta1 = 0.9;
			adam_beta2 = 0.999;
			adam_alpha = 1e-3;
			adam_sigma = 1e-8;
			adam_m.Reset(pfunc->GetFeatNum());
			adam_v.Reset(pfunc->GetFeatNum());
			adam_m.Fill(0);
			adam_v.Fill(0);
#endif
		}
		/// Run iteration. input the init-parameters.
		virtual bool Run(const double *pInitParams = NULL);
		/// Update the learning rate
		void UpdateGamma(int nIterNum);
		/// compute the update direction
		void UpdateDir(double *pDir, double *pGradient, const double *pParam);
		/// Update the parameters.
		virtual void Update(double *pdParam, const double *pdDir, double dStep);
		/// Print Information
		void PrintInfo();
	};

	
}