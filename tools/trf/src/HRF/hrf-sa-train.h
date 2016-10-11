#pragma once
#include "hrf-ml-train.h"
#include "hrf-corpus.h"
#include <omp.h>


namespace hrf
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
		int m_nMiniBatchTraining; ///< mini-batch for training set
		trf::CorpusRandSelect m_TrainSelect; ///< random select the sequence from corpus
		CorpusCache m_TrainCache; ///< cache all the h of training sequences.

		Vec<Prob> m_samplePi; ///< the length distribution used for sample

	private:
		Vec<double> m_vAllSampleLenCount; ///< the count of each length in all samples
		Vec<double> m_vCurSampleLenCount; ///< the count of length in samples of current iteration
		int m_nTotalSample; ///< the total sample number

		Vec<double> m_vEmpFeatExp; ///< the empirical expectation of features
		Vec<double> m_vEmpFeatVar; ///< the empirical variance of features
		
		Vec<double> m_vEmpExp; ///< the empirical expectation
		Vec<double> m_vEmpExp2; ///< the empirical expectation E[f^2]
		Vec<double> m_vSampleExp; ///< the sample expectation
		Vec<double> m_vSampleLen; ///< the sample length expectation

#ifndef _CD
		Array<ThreadData*> m_threadData; ///< save the last sequence of each threads
		Array<Seq*> m_aSeqs; ///< save the last sequence of each thread.
#endif
		//Array<trf::RandSeq<int>*> m_trainSelectPerLen; ///< save the index of training sequences of each length


		Mat<double> m_matEmpiricalExp; ///< the empirical expectation of each thread
		Mat<double> m_matEmpiricalExp2; ///< empirical E[f^2] of each thread
		Mat<double> m_matSampleExp; ///< the sample expectation of each thread
		Mat<double> m_matSampleLen; ///< the length count of sample of each thread

		/* for emprical variance estimation :*/
#ifdef _Var
		Vec<double> m_vExpValue; ///< the estimated E[f] at current iteration
		Vec<double> m_vExp2Value; ///< the estimated E[f^2] at current iteration
	public:
		double m_var_gap;
#endif
		//Vec<double> m_vEmpiricalVar; ///< current empirical variance E[f^2]-E[f]^2	

	public:
		AISConfig m_AISConfigForZ; ///< the AIS configuration for normalization
		AISConfig m_AISConfigForP; ///< the AIS configuration for calculating the LL.
		int m_nTrainHiddenSampleTimes; ///< the sample times for training sequence
		int m_nSampleHiddenSampleTimes; ///< the sample times for the hidden of samples
		int m_nCDSampleTimes; ///< the CD-n: the sample number.
		int m_nSASampleTimes; ///< the SA sample times
		
	public:
		File m_fdbg;  ///< output the sample pi/zete information
		File m_fparm; ///< output the parameters of each iteration
		File m_fgrad; ///< output the gradient of each iteration
		File m_fvar;  ///< output the variance at each iteration
		File m_fexp;  ///< output the expectation of each iteartion
		File m_fsamp; ///< output all the samples
		File m_ftrain;///< output all the training sequences
		File m_feat_mean; ///< output the empirical mean 
		File m_feat_var;  ///< output the empirical variance 

		bool m_bPrintTrain; ///< output the LL on training set
		bool m_bPrintValie; ///< output the LL on valid set
		bool m_bPrintTest;  ///< output the LL on test set

	public:
		SAfunc() :m_nMiniBatchSample(100), m_nMiniBatchTraining(100) {
#ifdef _Var
			m_var_gap = 1e-5;
#endif
			m_bPrintTrain = true;
			m_bPrintValie = true;
			m_bPrintTest = true;
		};
		SAfunc(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL, int nMinibatch = 100 )
		{
			Reset(pModel, pTrain, pValid, pTest, nMinibatch);
#ifdef _Var
			m_var_gap = 1e-5;
#endif
			m_bPrintTrain = true;
			m_bPrintValie = true;
			m_bPrintTest = true;
		}
		~SAfunc()
		{
			
#ifndef _CD
			for (int i = 0; i < m_aSeqs.GetNum(); i++)
				SAFE_DELETE(m_aSeqs[i]);
			for (int i = 0; i < m_threadData.GetNum(); i++) {
				SAFE_DELETE(m_threadData[i]);
			}
#endif
// 			for (int i = 0; i < m_trainSelectPerLen.GetNum(); i++) {
// 				SAFE_DELETE(m_trainSelectPerLen[i]);
// 			}
		}
		/// reset 
		void Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL, int nMinibatch = 100);
		/// print information
		void PrintInfo();
		/// get the ngram feature number
		int GetNgramFeatNum() const { return m_pModel->m_pFeat->GetNum(); }
		/// get the VH mat number
		int GetVHmatSize() const { return m_pModel->m_m3dVH.GetSize(); }
		/// get the CH mat number
		int GetCHmatSize() const { return m_pModel->m_m3dCH.GetSize(); }
		/// get the HH mat number
		int GetHHmatSize() const { return m_pModel->m_m3dHH.GetSize(); }
		/// get the bias mat number
		int GetBiasSize() const { return m_pModel->m_matBias.GetSize(); }
		/// get the nunber of all the weight up the exp
		int GetWeightNum() const { return m_pModel->GetParamNum(); }
		/// get the zeta parameter number
		int GetZetaNum() const { return m_pModel->GetMaxLen() + 1; }
		/// get a random sequence
		void RandSeq(Seq &seq, int nLen = -1);
		/// get the parameters
		void GetParam(double *pdParams);
		/// get the empirical variance of features
		//void GetFeatEmpVar(CorpusBase *pCorpus, Vec<double> &vVar);

		/// claculate the empirical expectation of features
		void GetEmpiricalFeatExp(Vec<double> &vExp);
		/// claculate the empirical variance of features
		void GetEmpiricalFeatVar(Vec<double> &vVar);

		/// calculate the empirical expectation of given sequence
		int GetEmpiricalExp(VecShell<double> &vExp, VecShell<double> &vExp2, Array<int> &aRandIdx);
		/// calculate the empirical expectation
		int GetEmpiricalExp(VecShell<double> &vExp, VecShell<double> &vExp2);
		/// calcualte the expectation of SA samples
		int GetSampleExp(VecShell<double> &vExp, VecShell<double> &vLen);
		/// perform CD process and get the expectation
		void PerfromCD(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen);
		/// perform SA process and get the expectation
		void PerfromSA(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen);
		/// perform SAMS, and then select the training sequences of the same length.
//		void PerfromSAMS(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen);
		/// Sample the most possible hidden and calculate the LL
		/* method = 0 : AIS,   =1 : Chib */
		double GetSampleLL(CorpusBase *pCorpus, int nCalNum = -1, int method = 0);
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
		double m_gamma_hidden;
		double m_gamma_zeta;

	public:

		LearningRate m_gain_lambda;
		LearningRate m_gain_hidden;
		LearningRate m_gain_zeta;

		bool m_bUpdate_lambda;
		bool m_bUpdate_zeta;

		double m_dir_gap;
		double m_zeta_gap;

		float m_fMomentum; ///< the momentum 
		int m_nAvgBeg; ///< if >0, then calculate the average

		double m_fEpochNum; ///< the current epoch number

		int m_nPrintPerIter;  ///< output the LL per iteration, if ==-1, the disable
		wb::Array<int> m_aWriteAtIter; ///< output temp model at some iteration

#ifdef _Var
		//double m_var_threshold;
		double m_gamma_var;
		LearningRate m_gain_var;
#endif
	public:
		SAtrain(SAfunc *pfunc = NULL) : Solve(pfunc)
		{
#ifndef _CD
			m_pAlgorithmName = "[SA]";
#else
			m_pAlgorithmName = "[CD]";
#endif

			m_gamma_lambda = 1;
			m_gamma_hidden = 1;
			m_gamma_zeta = 1;


			m_bUpdate_lambda = true;
			m_bUpdate_zeta = true;
			
			m_dir_gap = 0;
			m_zeta_gap = 0;

			m_fMomentum = 0;
			m_nAvgBeg = 0;

			m_fEpochNum = 0;
			m_nPrintPerIter = 1;
// #ifdef _Var
// 			m_var_threshold = 1e-4;
// #endif
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
		/// cut array
		int CutValue(double *p, int num, double gap);
	};

	
}