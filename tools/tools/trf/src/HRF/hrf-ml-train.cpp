#include "hrf-ml-train.h"
#include <omp.h>

namespace hrf
{
	MLfunc::MLfunc(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid /* = NULL */, CorpusBase *pTest /* = NULL */)
	{
		m_pathOutputModel = NULL;

		Reset(pModel, pTrain, pValid, pTest);
	}
	void MLfunc::Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid /* = NULL */, CorpusBase *pTest /* = NULL */)
	{
		m_pModel = pModel;
		m_pCorpusTrain = pTrain;
		m_pCorpusValid = pValid;
		m_pCorpusTest = pTest;

		if (m_pCorpusTrain) lout_variable(m_pCorpusTrain->GetNum());
		if (m_pCorpusValid) lout_variable(m_pCorpusValid->GetNum());
		if (m_pCorpusTest) lout_variable(m_pCorpusTest->GetNum());

		m_nParamNum = m_pModel->GetParamNum();
		m_values.Reset(m_nParamNum);

		/// Check maximum length
		int nMaxLen = m_pCorpusTrain->GetMaxLen();
		if (pValid)
			nMaxLen = max(nMaxLen, pValid->GetMaxLen());
		if (pTest)
			nMaxLen = max(nMaxLen, pTest->GetMaxLen());
		if (nMaxLen != m_pModel->m_maxlen) {
			lout_warning("[MLfunc] Reset: the max-len in training (" << nMaxLen
				<< ") is not equal to m_pModel->m_maxlen (" << m_pModel->m_maxlen<<")");
			lout_warning("[MLfunc] Reset: Re-set the model with length=" << nMaxLen);
			m_pModel->Reset(m_pModel->GetVocab(), m_pModel->m_hlayer, m_pModel->m_hnode, nMaxLen);
		}

		/// calculate the length distribution in training corpus
		Array<int> aLenCount;
		m_trainPi.Reset(nMaxLen+1);
		m_pCorpusTrain->GetLenCount(aLenCount);
		for (int i = 0; i < aLenCount.GetNum(); i++) {
			m_trainPi[i] = 1.0* aLenCount[i] / m_pCorpusTrain->GetNum();
		}
		m_pModel->SetPi(m_trainPi.GetBuf());

		lout_variable(nMaxLen);
		lout << "train-pi = [ "; lout.output(m_trainPi.GetBuf() + 1, m_trainPi.GetSize() - 1); lout << "]"<< endl;



// 		m_TrainSelect.Reset(m_pCorpusTrain);
// 		m_nMiniBatch = 10;
// 		m_nScanSeq = 0;
	}
	void MLfunc::SetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		for (int i = 0; i < m_nParamNum; i++) {
			m_values[i] = (PValue) pdParams[i];
		}
		m_pModel->SetParam(m_values.GetBuf());
		m_pModel->ExactNormalize();
	}
	void MLfunc::GetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		m_pModel->GetParam(m_values.GetBuf());
		for (int i = 0; i < m_nParamNum; i++) {
			pdParams[i] = m_values[i];
		}
	}
	double MLfunc::GetLL(CorpusBase *pCorpus, int nCalNum /* = -1 */)
	{
		int nThread = omp_get_max_threads();

		Array<VocabID> aSeq;
		Vec<double> vSum(nThread);
		Vec<int> vNum(nThread);
		vSum.Fill(0);
		vNum.Fill(0);

		Vec<double> vWeight(nThread);
		Vec<double> vLogz(nThread);
		Vec<double> vZeta(nThread);
		Vec<double> vPi(nThread);
		vWeight.Fill(0);
		vLogz.Fill(0);
		vZeta.Fill(0);
		vPi.Fill(0);
	

		int nCorpusNum = (nCalNum == -1) ? pCorpus->GetNum() : min(nCalNum, pCorpus->GetNum());
		Title::Precent(0, true, nCorpusNum-1, "omp GetLL");
#pragma omp parallel for firstprivate(aSeq)
		for (int i = 0; i < nCorpusNum; i++) {
			pCorpus->GetSeq(i, aSeq);

			VecShell<VocabID> x(aSeq.GetBuffer(), aSeq.GetNum());
			LogP logprob = m_pModel->GetLogProb(x);

			vSum[omp_get_thread_num()] += logprob;
			vNum[omp_get_thread_num()]++;
			
			int nLen = min(aSeq.GetNum(), m_pModel->GetMaxLen());
			vWeight[omp_get_thread_num()] += m_pModel->GetLogProb(x, false);
			vLogz[omp_get_thread_num()] += m_pModel->m_logz[nLen];
			vZeta[omp_get_thread_num()] += m_pModel->m_zeta[nLen];
			vPi[omp_get_thread_num()] += trf::Prob2LogP(m_pModel->m_pi[nLen]);


			Title::Precent();
		}

		double dsum = 0;
		int nNum = 0;
		for (int t = 0; t < nThread; t++) {
			dsum += vSum[t];
			nNum += vNum[t];

		}

		lout_variable(vSum.Sum() / nNum);
		lout_variable(vNum.Sum() / nNum);
		lout_variable(vLogz.Sum() / nNum);
		lout_variable(vZeta.Sum()/nNum);
		lout_variable(vPi.Sum()/nNum);

		lout_variable(m_pModel->m_logz[1]);
		lout_variable(vSum.Sum() / nNum);
		lout_variable((vWeight.Sum() - vZeta.Sum() + vPi.Sum()) / nNum);
		lout_variable((vWeight.Sum() - vLogz.Sum() + vPi.Sum()) / nNum);

		return dsum / nNum;
	}
	double MLfunc::GetValue()
	{
		//SetParam(pdParams);
		
		return -GetLL(m_pCorpusTrain);

		return 0;
	}
	void MLfunc::GetGradient(double *pdGradient)
	{
		//SetParam(pdParams);

		Vec<double> aExpEmpirical(m_nParamNum);
		Vec<double> aExpGivenX(m_nParamNum);
		Vec<double> aExpTheoretical(m_nParamNum);

		aExpEmpirical.Fill(0);
		Array<VocabID> aSeq;
		int nNum = 0;
		for (int i = 0; i < m_pCorpusTrain->GetNum(); i++) {
			m_pCorpusTrain->GetSeq(i, aSeq);

			VecShell<VocabID> x(aSeq.GetBuffer(), aSeq.GetNum());
			aExpGivenX.Fill(0);
			m_pModel->GetHiddenExp(x, aExpGivenX.GetBuf());
			aExpEmpirical += aExpGivenX;
			nNum++;
		}
		aExpEmpirical *= 1.0 / nNum;

		m_pModel->GetNodeExp(aExpTheoretical.GetBuf(), m_trainPi.GetBuf());
		
		for (int i = 0; i < m_nParamNum; i++) {
			pdGradient[i] = -(aExpEmpirical[i] - aExpTheoretical[i]);
		}


		static File fileDbg("GradientML.dbg", "wt");
		VecShell<double> featexp;
		Mat3dShell<double> VHexp, CHexp, HHexp;
		MatShell<double> Bexp;
		m_pModel->BufMap(pdGradient, featexp, VHexp, CHexp, HHexp, Bexp);
		fileDbg.PrintArray("%f ", featexp.GetBuf(), featexp.GetSize());
		fileDbg.PrintArray("%f ", VHexp.GetBuf(), VHexp.GetSize());
		fileDbg.PrintArray("%f ", HHexp.GetBuf(), HHexp.GetSize());
		fileDbg.Print("\n");
		fileDbg.PrintArray("%f ", aExpEmpirical.GetBuf(), m_nParamNum);
		fileDbg.PrintArray("%f ", aExpTheoretical.GetBuf(), m_nParamNum);
/*		fileDbg.PrintArray("%f ", pdGradient, m_nParamNum);*/
		//Pause();
		/*return false;*/
	}
	int MLfunc::GetExtraValues(int t/*, double *pdParams*/, double *pdValues)
	{
		//SetParam(pdParams);

		if ( (t - 1) % 10 == 0) {
			m_pModel->WriteT(m_pathOutputModel);
		}

		int nValue = 0;
		pdValues[nValue++] = -GetLL(m_pCorpusTrain);
		if (m_pCorpusValid) {
			pdValues[nValue++] = -GetLL(m_pCorpusValid);
		}
		if (m_pCorpusTest) {
			pdValues[nValue++] = -GetLL(m_pCorpusTest);
		}

		return nValue;

	}
}