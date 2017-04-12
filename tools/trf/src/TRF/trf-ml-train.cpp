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


#include "trf-ml-train.h"
#include <omp.h>

namespace trf
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
	

		/// Check maximum length
		int nMaxLen = m_pCorpusTrain->GetMaxLen();
		if (pValid)
			nMaxLen = max(nMaxLen, pValid->GetMaxLen());
		if (pTest)
			nMaxLen = max(nMaxLen, pTest->GetMaxLen());

		if (m_pModel->GetMaxLen() <= 0) {
			lout_warning("[MLfunc] Reset: Re-set the model with length=" << nMaxLen);
			const char* strNote = 
				"------------  [ Note ] --------------\n"
				"As the inital len is 0, then set len to the max-len of corpus\n"
				"1. If the empirical pi is too small, please turn down the len manully\n"
				"2. If the NLL is unreasonablly large, this is because that the pi for some length is to small (close to zero),"
				"please turn down the len manully";
			lout << strNote << endl;
			m_pModel->Reset(m_pModel->GetVocab(), nMaxLen);
		}
		else if (nMaxLen != m_pModel->m_maxlen) {
			lout_warning("[MLfunc] Reset: the max-len in training (" << nMaxLen
				<< ") is not equal to m_pModel->m_maxlen (" << m_pModel->m_maxlen<<")");
		}

		/// calculate the length distribution in training corpus
		Array<int> aLenCount;
		m_trainPi.Reset(m_pModel->GetMaxLen()+1);
		m_trainPi.Fill(0);
		m_pCorpusTrain->GetLenCount(aLenCount);
		for (int i = 1; i < aLenCount.GetNum(); i++) {
			int nLen = min(m_pModel->GetMaxLen(), i);
			m_trainPi[nLen] += aLenCount[i];
		}
		m_trainPi /= m_pCorpusTrain->GetNum();
		m_pModel->SetPi(m_trainPi.GetBuf());

		lout_variable(nMaxLen);
		lout_variable(m_pModel->GetMaxLen());
		lout << "train-pi = [ "; lout.output(m_trainPi.GetBuf() + 1, m_trainPi.GetSize() - 1); lout << "]"<< endl;


		/// get empirical expectation
		GetEmpExp(m_pCorpusTrain, m_vEmpiricalExp);
	}
	void MLfunc::SetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		m_value.Reset(m_nParamNum);
		for (int i = 0; i < m_nParamNum; i++)
			m_value[i] = (PValue)pdParams[i];
		m_pModel->SetParam(m_value.GetBuf());

		m_pModel->ExactNormalize();
	}
	void MLfunc::GetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		m_value.Reset(m_nParamNum);
		m_pModel->GetParam(m_value.GetBuf());
		
		for (int i = 0; i < m_nParamNum; i++)
			pdParams[i] = m_value[i];
	}
	double MLfunc::GetLL(CorpusBase *pCorpus, int nCalNum /* = -1 */, Vec<double> *pLL /* = NULL */)
	{
		int nThread = omp_get_max_threads();

		Array<VocabID> aSeq;
		Vec<double> vSum(nThread);
		Vec<int> vNum(nThread);
		vSum.Fill(0);
		vNum.Fill(0);

		int nCorpusNum = (nCalNum == -1) ? pCorpus->GetNum() : min(nCalNum, pCorpus->GetNum());

		if (pLL)
			pLL->Reset(nCorpusNum);

		//lout.Progress(0, true, nCorpusNum-1, "[MLfunc] LL:");
#pragma omp parallel for firstprivate(aSeq)
		for (int i = 0; i < nCorpusNum; i++) {
			pCorpus->GetSeq(i, aSeq);

			Seq seq;
			seq.Set(aSeq, m_pModel->m_pVocab);
			LogP logprob = m_pModel->GetLogProb(seq);

			vSum[omp_get_thread_num()] += logprob;
			vNum[omp_get_thread_num()]++;
			if (pLL)
				(*pLL)[i] = logprob;

// #pragma omp critical 
// 			{
// 				lout.Progress();
// 			}
			
		}

		double dsum = 0;
		int nNum = 0;
		for (int t = 0; t < nThread; t++) {
			dsum += vSum[t];
			nNum += vNum[t];
		}
		return dsum / nNum;
	}
	void MLfunc::GetEmpExp(CorpusBase *pCorpus, Vec<double> &vExp)
	{
		Array<VocabID> aSeq;
		Mat<double> matExp(omp_get_max_threads(), m_nParamNum);
		matExp.Fill(0);

		lout.Progress(0, true, pCorpus->GetNum()-1, "[MLfunc] E[f]  :");
#pragma omp parallel for firstprivate(aSeq)
		for (int i = 0; i < pCorpus->GetNum(); i++) {
			pCorpus->GetSeq(i, aSeq);

			Seq seq;
			seq.Set(aSeq, m_pModel->m_pVocab);
			m_pModel->FeatCount(seq, matExp[omp_get_thread_num()].GetBuf());

#pragma omp critical 
			{
				lout.Progress();
			}
			
		}

		vExp.Reset(m_nParamNum);
		vExp.Fill(0);
		for (int t = 0; t < omp_get_max_threads(); t++) {
			vExp += matExp[t];
		}
		vExp /= pCorpus->GetNum();
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
		Vec<double> aExpTheoretical(m_nParamNum);


		m_pModel->GetNodeExp(aExpTheoretical.GetBuf(), m_trainPi.GetBuf());
		
		for (int i = 0; i < m_nParamNum; i++) {
			pdGradient[i] = -(m_vEmpiricalExp[i] - aExpTheoretical[i]);
		}


		static File fileDbg("GradientML.dbg", "wt");
		fileDbg.PrintArray("%f ", m_vEmpiricalExp.GetBuf(), m_nParamNum);
		fileDbg.PrintArray("%f ", aExpTheoretical.GetBuf(), m_nParamNum);
	}
	int MLfunc::GetExtraValues(int t/*, double *pdParams*/, double *pdValues)
	{
		//SetParam(pdParams);

		if ( (t - 1) % 10 == 0) {
			m_pModel->WriteT(m_pathOutputModel);
		}

		int nValue = 0;
		pdValues[nValue++] = -GetLL(m_pCorpusTrain);
		if (m_pCorpusValid) pdValues[nValue++] = -GetLL(m_pCorpusValid);
		if (m_pCorpusTest)  pdValues[nValue++] = -GetLL(m_pCorpusTest);

		return nValue;

	}
}