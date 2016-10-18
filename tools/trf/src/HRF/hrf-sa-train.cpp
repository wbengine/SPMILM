#include "hrf-sa-train.h"
#include "wb-log.h"

namespace hrf
{
	ThreadData::~ThreadData()
	{
		for (int i = 0; i < aSeqs.GetNum(); i++) {
			SAFE_DELETE(aSeqs[i]);
		}
	}
	void ThreadData::Create(int maxlen, Model *pModel)
	{
		aSeqs.SetNum(maxlen + 1);
		aSeqs.Fill(NULL);
		for (int i = 1; i < aSeqs.GetNum(); i++) {
			aSeqs[i] = new Seq;
			pModel->RandSeq(*aSeqs[i], i);
		}
	}

	void SAfunc::Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid /* = NULL */, CorpusBase *pTest /* = NULL */, int nMinibatch /* = 100 */)
	{
		MLfunc::Reset(pModel, pTrain, pValid, pTest);
		m_nMiniBatchSample = nMinibatch;
		m_nMiniBatchTraining = nMinibatch;
		m_TrainSelect.Reset(pTrain);
		m_TrainCache.Reset(pTrain, pModel);
		/*
		sampling pi
		*/
		m_samplePi.Copy(m_trainPi);

		lout << "Smoothing the pi" << endl;
		double dMax = 0;
		int iMax = 0;
		for (int i = 1; i < m_trainPi.GetSize(); i++) {
			if (m_trainPi[i] > dMax) {
				dMax = m_trainPi[i];
				iMax = i;
			}
		}
		for (int i = 1; i < iMax; i++) {
			m_samplePi[i] = dMax;
		}
		for (int i = 1; i < m_samplePi.GetSize(); i++) {
			m_samplePi[i] = max((double)m_samplePi[i], 1e-5);
		}
		trf::LineNormalize(m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1);

		lout << "sample-pi = [ "; lout.output(m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1); lout << "]" << endl;
		m_pModel->SetPi(m_samplePi.GetBuf());

		/* save the sample count */
		m_vAllSampleLenCount.Reset(m_pModel->GetMaxLen()+1);
		m_vCurSampleLenCount.Reset(m_pModel->GetMaxLen() + 1);
		m_vAllSampleLenCount.Fill(0);
		m_nTotalSample = 0;

		/* for SA estimateio. there are two set of paremeters
		    i.e. feature weight \lambda and normalization constants \zeta
		*/
		m_nParamNum = m_pModel->GetParamNum() + m_pModel->GetMaxLen() + 1;
#ifdef _Var
		/* set the var as part of the paremeters */
		/* only record the var for hidden-dependent parameters */
		int nHiddenParamNum = m_pModel->GetParamNum() - m_pModel->m_pFeat->GetNum();
		m_nParamNum += nHiddenParamNum * 2;
		m_vExpValue.Reset(nHiddenParamNum);
		m_vExp2Value.Reset(nHiddenParamNum);
		m_vExpValue.Fill(0);
		m_vExp2Value.Fill(1);
// 		m_vEstimatedVar.Reset(m_pModel->GetParamNum());
// 		m_vEstimatedVar.Fill(1);
#endif

		m_nTrainHiddenSampleTimes = 1;
		m_nSampleHiddenSampleTimes = 1;
		m_nCDSampleTimes = 1;
		m_nSASampleTimes = 1;
		
		// count feature expectation and variance
		GetEmpiricalFeatExp(m_vEmpFeatExp);
		GetEmpiricalFeatVar(m_vEmpFeatVar);

	}
	void SAfunc::PrintInfo()
	{
		lout << "[SAfunc] *** Info: *** " << endl;
		lout << "  "; lout_variable(m_nMiniBatchTraining);
		lout << "  "; lout_variable(m_nMiniBatchSample);
		lout << "  "; lout_variable(m_nTrainHiddenSampleTimes);
		lout << "  "; lout_variable(m_nSampleHiddenSampleTimes);
		lout << "  "; lout_variable(m_nCDSampleTimes);
		lout << "  "; lout_variable(m_nSASampleTimes);
#ifdef _Var
		lout << "  "; lout_variable(m_var_gap);
#endif
		lout << "  "; lout_variable(m_bSAMSSample);
		lout << "  [AISConfig for Z]  nChain=" << m_AISConfigForZ.nChain << " nIter=" << m_AISConfigForZ.nInter << endl;
		lout << "  [AISConfig for LL] nChain=" << m_AISConfigForP.nChain << " nIter=" << m_AISConfigForP.nInter << endl;
		lout << "[SAfunc] *** [End] ***" << endl;
	}
	void SAfunc::RandSeq(Seq &seq, int nLen /* = -1 */)
	{
		m_pModel->RandSeq(seq, nLen);
	}
	void SAfunc::SetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		/* set lambda */
		for (int i = 0; i < m_pModel->GetParamNum(); i++) {
			m_values[i] = (PValue)pdParams[i];
		}
		m_pModel->SetParam(m_values.GetBuf());
		m_pModel->ExactNormalize(1); // only calculate Z_1

		/* set zeta */
		m_pModel->SetZeta(pdParams + m_pModel->GetParamNum());

#ifdef _Var
		/* set var */
		double *p = pdParams + GetWeightNum() + GetZetaNum();
		int nVarNum = m_vExpValue.GetSize();
		m_vExpValue.Copy( VecShell<double>(p, nVarNum));
		m_vExp2Value.Copy(VecShell<double>(p + nVarNum, nVarNum));
#endif
		if (m_fparm.Good()) {
			m_fparm.PrintArray("%f ", pdParams, m_nParamNum);
		}
	}
	void SAfunc::GetParam(double *pdParams)
	{
		if (pdParams == NULL)
			return;

		/* get lambda */
		m_values.Reset(m_pModel->GetParamNum());
		m_pModel->GetParam(m_values.GetBuf());
		for (int i = 0; i < m_pModel->GetParamNum(); i++) {
			pdParams[i] = m_values[i];
		}
		/* get zeta */
		pdParams += m_pModel->GetParamNum();
		for (int i = 0; i <= m_pModel->GetMaxLen(); i++) {
			pdParams[i] = m_pModel->m_zeta[i];
		}
#ifdef _Var
		/* get var */
		pdParams += GetZetaNum();
		for (int i = 0; i < m_vExpValue.GetSize(); i++) {
			*pdParams = m_vExpValue[i];
			pdParams++;
		}
		for (int i = 0; i < m_vExp2Value.GetSize(); i++) {
			*pdParams = m_vExp2Value[i];
			pdParams++;
		}
#endif

	}

	void SAfunc::GetEmpiricalFeatExp(Vec<double> &vExp)
	{
		/* for empirical exp */
		Array<VocabID> aSeq;
		int nFeat = m_pModel->m_pFeat->GetNum();
		vExp.Reset(nFeat);
		m_matEmpiricalExp.Reset(omp_get_max_threads(), nFeat);
		m_matEmpiricalExp.Fill(0);

		lout.Progress(0, true, m_pCorpusTrain->GetNum()-1, "[SAfunc] E[f]  :");
#pragma omp parallel for firstprivate(aSeq)
		for (int i = 0; i < m_pCorpusTrain->GetNum(); i++) {
			m_pCorpusTrain->GetSeq(i, aSeq);
			trf::Seq trfseq;
			trfseq.Set(aSeq, m_pModel->GetVocab());
			((trf::Model*)m_pModel)->FeatCount(trfseq, m_matEmpiricalExp[omp_get_thread_num()].GetBuf());
#pragma omp critical
			lout.Progress();
			//lout.output(m_matEmpiricalExp[omp_get_thread_num()].GetBuf() + m_pModel->m_pFeat->GetNum(), 10);
		}

		vExp.Fill(0);
		for (int t = 0; t < omp_get_max_threads(); t++) {
			vExp += m_matEmpiricalExp[t]; // E[f]
		}
		vExp /= m_pCorpusTrain->GetNum(); // E[f]

		if (m_feat_mean.Good()) {
			lout << "Write Empirical Mean ..." << endl;
			Vec<PValue> aLogExp(vExp.GetSize());
			for (int i = 0; i < aLogExp.GetSize(); i++) aLogExp[i] = log(vExp[i]);
			m_pModel->m_pFeat->WriteT(m_feat_mean, aLogExp.GetBuf());
		}
	}
	void SAfunc::GetEmpiricalFeatVar(Vec<double> &vVar)
	{
		int nThread = omp_get_max_threads();
		Prob *pi = m_trainPi.GetBuf();
		CorpusBase *pCorpus = m_pCorpusTrain;
		int nFeatNum = m_pModel->m_pFeat->GetNum();

		vVar.Reset(nFeatNum);
		vVar.Fill(0);
		Array<VocabID> aSeq;
		Vec<double> vExpf2(nFeatNum);
		Vec<double> vExp_l(nFeatNum);

		Mat<double> matExpf2(nThread, vExpf2.GetSize());
		Mat<double> matExp_l(nThread, vExp_l.GetSize());

		vExpf2.Fill(0);
		vExp_l.Fill(0);
		matExpf2.Fill(0);
		matExp_l.Fill(0);

		/// Count p[f^2]
		lout.Progress(0, true, pCorpus->GetNum() - 1, "[SAfunc] E[f^2]:");
#pragma omp parallel for firstprivate(aSeq)
		for (int l = 0; l < pCorpus->GetNum(); l++) {
			double *pExpf2 = matExpf2[omp_get_thread_num()].GetBuf();
			pCorpus->GetSeq(l, aSeq);
			trf::Seq seq;
			seq.Set(aSeq, m_pModel->m_pVocab);

			int nLen = min(m_pModel->GetMaxLen(), seq.GetLen());

			LHash<int, int> aFeatNum;
			bool bFound;
			Array<int> afeat;
			m_pModel->m_pFeat->Find(afeat, seq);
			for (int i = 0; i < afeat.GetNum(); i++) {
				int *p = aFeatNum.Insert(afeat[i], bFound);
				if (!bFound) *p = 0;
				(*p) += 1;
			}
			LHashIter<int, int> iter(&aFeatNum);
			int *pCount;
			int nFeat;
			while (pCount = iter.Next(nFeat)) {
				pExpf2[nFeat] += pow((double)(*pCount), 2);
			}
#pragma omp critical 
			lout.Progress();
		}

		vExpf2.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vExpf2 += matExpf2[t];
		}
		vExpf2 /= pCorpus->GetNum();


		//lout_variable(aExpFeatSqu[38272]);

		/// Count p_l[f]
		/// As save p_l[f] for all the length cost too much memory. So we calculate each p_l[f] separately.
		lout.Progress(0, true, m_pModel->GetMaxLen(), "[SAfunc] E_l[f]:");
		for (int nLen = 1; nLen <= m_pModel->GetMaxLen(); nLen++)
		{
			matExp_l.Fill(0);

			Array<int> aSeqId;
			/// find all the sequence with length nLen
			for (int i = 0; i < pCorpus->GetNum(); i++) {
				pCorpus->GetSeq(i, aSeq);
				int nSeqLen = aSeq.GetNum();
				if (nLen == m_pModel->GetMaxLen()) {
					if (nSeqLen < nLen)
						continue;
				}
				else {
					if (nSeqLen != nLen)
						continue;
				}
				aSeqId.Add(i);
			}

#pragma omp parallel for firstprivate(aSeq)
			for (int k = 0; k < aSeqId.GetNum(); k++)
			{
				pCorpus->GetSeq(aSeqId[k], aSeq);

				trf::Seq seq;
				seq.Set(aSeq, m_pModel->m_pVocab);
				((trf::Model*)m_pModel)->FeatCount(seq, matExp_l[omp_get_thread_num()].GetBuf());
			}

			if (aSeqId.GetNum() > 0) {
				vExp_l.Fill(0);
				for (int t = 0; t < nThread; t++) {
					vExp_l += matExp_l[t];
				}
				vExp_l /= aSeqId.GetNum();
			}
			else {
				vExp_l.Fill(0);
			}


			for (int i = 0; i < nFeatNum; i++)
				vExpf2[i] -= pi[nLen] * pow(vExp_l[i], 2);  /// calcualte p[f^2] - \pi_l * p_l[f]^2

			lout.Progress(nLen);
		}

		/// output the zero number
		int nZero = 0;
		for (int i = 0; i < nFeatNum; i++) {
			if (vExpf2[i] == 0)
				nZero++;
		}
		if (nZero > 0) {
			lout_warning("[EmpiricalVar] Exist zero expectation  (zero-num=" << nZero << ")");
		}


		///save
		vVar = vExpf2;

		// Write
		if (m_feat_var.Good()) {
			lout << "Write Empirical Var ..." << endl;
			Vec<PValue> aLogVar(vVar.GetSize());
			for (int i = 0; i < aLogVar.GetSize(); i++) aLogVar[i] = log(vVar[i]);
			m_pModel->m_pFeat->WriteT(m_feat_var, aLogVar.GetBuf());
		}
	}
	int SAfunc::GetEmpiricalExp(VecShell<double> &vExp, VecShell<double> &vExp2, Array<int> &aRandIdx)
	{
		int nThread = omp_get_max_threads();
		/* for empirical exp */
		m_matEmpiricalExp.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp.Fill(0);

		/*for empirical variance estimation */
		m_matEmpiricalExp2.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp2.Fill(0);

		Vec<int> vTotalLen(nThread);
		vTotalLen.Fill(0);

		/* count the empirical expectation */
#pragma omp parallel for
		for (int i = 0; i < aRandIdx.GetNum(); i++) {

			int tnum = omp_get_thread_num();
			Vec<double> vExpGivenX(m_pModel->GetParamNum());
			vExpGivenX.Fill(0);

			Seq *pSeq = m_TrainCache.GetSeq(aRandIdx[i]);
			int nLen = pSeq->GetLen();

			/* sample H */
			for (int j = 0; j < m_nTrainHiddenSampleTimes; j++) {
				m_pModel->SampleHAndCGivenX(*pSeq); /// several times sampling
			}

			//m_pModel->GetHiddenExp(VecShell<VocabID>(aSeq.GetBuffer(), nLen), vExpGivenX.GetBuf());
			m_pModel->FeatCount(*pSeq, vExpGivenX); // count

			m_matEmpiricalExp[tnum] += vExpGivenX;
			for (int n = 0; n < vExpGivenX.GetSize(); n++) {
				m_matEmpiricalExp2[tnum][n] += pow(vExpGivenX[n], 2);
			}
			vTotalLen[tnum] += nLen;


			if (m_ftrain.Good()) {
#pragma omp critical
				{
					pSeq->Write(m_ftrain);
				}
			}

		}


		// only change the hidden depended value
		vExp.Fill(0);
		vExp2.Fill(0);
		int nTotalLen = 0;
		for (int t = 0; t < nThread; t++) {
			vExp += m_matEmpiricalExp[t]; // E[f]
			vExp2 += m_matEmpiricalExp2[t]; // E[f^2]
			nTotalLen += vTotalLen[t];
		}
		vExp /= m_nMiniBatchTraining;
		vExp2 /= m_nMiniBatchTraining;

		return nTotalLen;
	}
	int SAfunc::GetEmpiricalExp(VecShell<double> &vExp, VecShell<double> &vExp2)
	{
		//Array<VocabID> aSeq;
		Array<int> aRandIdx;
		aRandIdx.SetNum(m_nMiniBatchTraining);
		m_TrainSelect.GetIdx(aRandIdx.GetBuffer(), aRandIdx.GetNum());

		return GetEmpiricalExp(vExp, vExp2, aRandIdx);
	}
	int SAfunc::GetSampleExp(VecShell<double> &vExp, VecShell<double> &vLen)
	{
		int nThread = omp_get_max_threads();
		m_matSampleExp.Reset(nThread, m_pModel->GetParamNum());
		m_matSampleLen.Reset(nThread, m_pModel->GetMaxLen() + 1);

		m_matSampleExp.Fill(0);
		m_matSampleLen.Fill(0);

		Vec<int> vTotalLen(nThread);
		vTotalLen.Fill(0);


		// init the sequence
		if (m_aSeqs.GetNum() != nThread) {
			for (int i = 0; i < nThread; i++) {
				m_aSeqs[i] = new Seq;
				m_pModel->RandSeq(*m_aSeqs[i]);
			}
		}

		/* sampling */
#pragma omp parallel for
		for (int sample = 0; sample < m_nMiniBatchSample; sample++)
		{
			Vec<double> vExpGivenX(m_pModel->GetParamNum());
			vExpGivenX.Fill(0);

			int tid = omp_get_thread_num();
			m_pModel->Sample(*m_aSeqs[tid]);
			int nLen = min(m_pModel->GetMaxLen(), m_aSeqs[tid]->GetLen());

			/* sample hidden several times */
			for (int j = 0; j < m_nSampleHiddenSampleTimes; j++) {
				m_pModel->SampleHAndCGivenX(*m_aSeqs[tid]);     ///< sample hidden
			}

// 			m_pModel->GetHiddenExp(m_aSeqs[tid]->GetWordSeq(), vExpGivenX.GetBuf());
// 			vExpGivenX *= m_trainPi[nLen] / m_pModel->m_pi[nLen];
// 			m_matSampleExp[tid] += vExpGivenX;
			m_pModel->FeatCount(*m_aSeqs[tid], m_matSampleExp[tid], m_trainPi[nLen] / m_pModel->m_pi[nLen]);
			m_matSampleLen[tid][nLen]++;
			vTotalLen[tid] += m_aSeqs[tid]->GetLen();


			if (m_fsamp.Good()) {
#pragma omp critical
				{
					m_aSeqs[tid]->Write(m_fsamp);
				}
			}
			
		}
		lout << " len-jump acc-rate=";
		lout_variable_rate(m_pModel->m_nLenJumpAccTimes, m_pModel->m_nLenJumpTotalTime);
		m_pModel->m_nLenJumpAccTimes = 0;
		m_pModel->m_nLenJumpTotalTime = 0;
		lout << endl;



		// summarization
		vExp.Fill(0);
		vLen.Fill(0);
		int nTotalLen = 0;
		for (int t = 0; t < nThread; t++) {
			vExp += m_matSampleExp[t];
			vLen += m_matSampleLen[t];
			nTotalLen += vTotalLen[t];
		}
		m_vAllSampleLenCount += vLen; /// save the length count
		m_vCurSampleLenCount.Copy(vLen); /// save current length count
		m_nTotalSample += m_nMiniBatchSample;

		vExp /= m_nMiniBatchSample;
		vLen /= m_nMiniBatchSample;

		return nTotalLen;
	}

	void SAfunc::PerfromCD(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen)
	{
		int nThread = omp_get_max_threads();
		/* for empirical expectation p[f] */
		m_matEmpiricalExp.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp.Fill(0);

		/*for empirical variance estimation p[f^2] */
		m_matEmpiricalExp2.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp2.Fill(0);

		/* for sample expectation p_n[f] */
		m_matSampleExp.Reset(nThread, m_pModel->GetParamNum());
		m_matSampleExp.Fill(0);

		/* for the length */
		m_matSampleLen.Reset(nThread, m_pModel->GetMaxLen() + 1);
		m_matSampleLen.Fill(0);

		Array<VocabID> aSeq;
		Vec<int> aRanIdx(m_nMiniBatchTraining);
		m_TrainSelect.GetIdx(aRanIdx.GetBuf(), m_nMiniBatchTraining);

		/* count the empirical variance */
#pragma omp parallel for firstprivate(aSeq) //保证aSeq是每个线程独立变量
		for (int i = 0; i < m_nMiniBatchTraining; i++) {

			int tnum = omp_get_thread_num();
			Vec<double> vExpGivenX(m_pModel->GetParamNum());
			vExpGivenX.Fill(0);

			/* read a sequence*/
			Array<int> aSeq;
			m_pCorpusTrain->GetSeq(aRanIdx[i], aSeq);
			int nLen = aSeq.GetNum();
			

			/* empirical expectation */
			m_pModel->GetHiddenExp(VecShell<int>(aSeq, nLen), vExpGivenX.GetBuf());
			m_matEmpiricalExp[tnum] += vExpGivenX;
			for (int n = 0; n < vExpGivenX.GetSize(); n++) {
				m_matEmpiricalExp2[tnum][n] += pow(vExpGivenX[n], 2);
			}

			if (m_ftrain.Good()) {
				m_ftrain.PrintArray("%d ", aSeq.GetBuffer(), nLen);
			}

			/* sample X and then sample H again */
			Seq seq;
			m_pModel->RandSeq(seq, nLen);
			seq.x.Set(aSeq, m_pModel->GetVocab());
			/* perform n times samples */
			for (int j = 0; j < m_nCDSampleTimes; j++) {
				for (int nPos = 0; nPos < nLen; nPos++) {
					m_pModel->SampleC(seq, nPos);
					m_pModel->SampleW(seq, nPos);
				}
				m_pModel->SampleHAndCGivenX(seq);
			}

			/* sample expectation */
			m_pModel->FeatCount(seq, m_matSampleExp[tnum]);
			m_matSampleLen[tnum][nLen]++;

			if (m_fsamp.Good()) {
				seq.Write(m_fsamp);
			}
			

			//Title::Precent();
		}

		// summarization
		vEmpExp.Fill(0);
		vEmpExp2.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vEmpExp += m_matEmpiricalExp[t]; // E[f]
			vEmpExp2 += m_matEmpiricalExp2[t]; // E[f^2]
		}
		vEmpExp /= m_nMiniBatchTraining; // E[f]
		vEmpExp2 /= m_nMiniBatchTraining; // E[f^2]


		// summarization
		vSamExp.Fill(0);
		vLen.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vSamExp += m_matSampleExp[t];
			vLen += m_matSampleLen[t];
		}
		m_vAllSampleLenCount += vLen; /// save the length count
		m_vCurSampleLenCount.Copy(vLen); /// save current length count
		m_nTotalSample += m_nMiniBatchTraining;

		vSamExp /= m_nMiniBatchTraining;
		vLen /= m_nMiniBatchTraining;

	}
	void SAfunc::PerfromSA(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen)
	{
		lout_assert(m_nMiniBatchSample == m_nMiniBatchTraining);

		int nThread = omp_get_max_threads();
		/* for empirical expectation p[f] */
		m_matEmpiricalExp.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp.Fill(0);

		/*for empirical variance estimation p[f^2] */
		m_matEmpiricalExp2.Reset(nThread, m_pModel->GetParamNum());
		m_matEmpiricalExp2.Fill(0);

		/* for sample expectation p_n[f] */
		m_matSampleExp.Reset(nThread, m_pModel->GetParamNum());
		m_matSampleExp.Fill(0);

		/* for the length */
		m_matSampleLen.Reset(nThread, m_pModel->GetMaxLen() + 1);
		m_matSampleLen.Fill(0);

		//Array<VocabID> aSeq;
		Vec<int> aRanIdx(m_nMiniBatchTraining);
		Vec<int> aRanLen(m_nMiniBatchTraining);
		m_TrainSelect.GetIdx(aRanIdx.GetBuf(), m_nMiniBatchTraining);

		/* count the empirical variance */
#pragma omp parallel for firstprivate(aSeq) 
		for (int i = 0; i < m_nMiniBatchTraining; i++) {

			int tnum = omp_get_thread_num();
			Vec<double> vExpGivenX(m_pModel->GetParamNum());
			vExpGivenX.Fill(0);

			/* read a sequence*/
			Seq *pSeq = m_TrainCache.GetSeq(aRanIdx[i]);
			int nLen = pSeq->GetLen();
			aRanLen[i] = nLen; /// record the length of the training sequence

			/* sample H */
			for (int j = 0; j < m_nTrainHiddenSampleTimes; j++) {
				m_pModel->SampleHAndCGivenX(*pSeq); /// several times sampling
			}

			/* empirical expectation */
			m_pModel->FeatCount(*pSeq, vExpGivenX); /// count 
			//m_pModel->GetHiddenExp(VecShell<int>(aSeq, nLen), vExpGivenX.GetBuf()); /// count 
			m_matEmpiricalExp[tnum] += vExpGivenX;
			for (int n = 0; n < vExpGivenX.GetSize(); n++) {
				m_matEmpiricalExp2[tnum][n] += pow(vExpGivenX[n], 2);
			}

			if (m_ftrain.Good()) {
#pragma omp critical
				{
					pSeq->Write(m_ftrain);
				}
			}
		}

		// init the sequence
		if (m_threadData.GetNum() != nThread) {
			m_threadData.SetNum(nThread);
			for (int i = 0; i < m_threadData.GetNum(); i++) {
				m_threadData[i] = new ThreadData;
				m_threadData[i]->Create(m_pModel->GetMaxLen(), m_pModel);
			}
		}

		/* SA sampling */
#pragma omp parallel for
		for (int i = 0; i < m_nMiniBatchTraining; i++)
		{
			int threadID = omp_get_thread_num();

			/* sample a length */
			int nLen = aRanLen[i];
			lout_assert(nLen >= 1);
			lout_assert(nLen <= m_pModel->GetMaxLen());

			/* perform gibbs */
			Seq *pSeq = m_threadData[threadID]->aSeqs[nLen];
			for (int j = 0; j < m_nSASampleTimes; j++)
				m_pModel->MarkovMove(*pSeq);
				//m_pModel->Sample(*pSeq);

			/* sample hidden several times */
			for (int j = 0; j < m_nSampleHiddenSampleTimes; j++) {
				m_pModel->SampleHAndCGivenX(*pSeq);     ///< sample hidden
			}

			/* expectation */
			m_pModel->FeatCount(*pSeq, m_matSampleExp[threadID]);
			//m_pModel->GetHiddenExp(pSeq->GetWordSeq(), m_matSampleExp[threadID].GetBuf());
			m_matSampleLen[threadID][nLen]++;

			if (m_fsamp.Good()) {
#pragma omp critical 
				{
					pSeq->Write(m_fsamp);
				}
			}

		}

		// summarization
		vEmpExp.Fill(0);
		vEmpExp2.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vEmpExp += m_matEmpiricalExp[t]; // E[f]
			vEmpExp2 += m_matEmpiricalExp2[t]; // E[f^2]
		}
		vEmpExp /= m_nMiniBatchTraining; // E[f]
		vEmpExp2 /= m_nMiniBatchTraining; // E[f^2]

		// summarization
		vSamExp.Fill(0);
		vLen.Fill(0);
		for (int t = 0; t < nThread; t++) {
			vSamExp += m_matSampleExp[t];
			vLen += m_matSampleLen[t];
		}
		m_vAllSampleLenCount += vLen; /// save the length count
		m_vCurSampleLenCount.Copy(vLen); /// save current length count
		m_nTotalSample += m_nMiniBatchTraining;

		vSamExp /= m_nMiniBatchTraining;
		vLen /= m_nMiniBatchTraining;

	}
// 	void SAfunc::PerfromSAMS(VecShell<double> &vEmpExp, VecShell<double> &vSamExp, VecShell<double> &vEmpExp2, VecShell<double> &vLen)
// 	{
// 		lout_assert(m_nMiniBatchTraining == m_nMiniBatchSample);
// 		/// perform SAMS
// 		int nTotalSampleLen = GetSampleExp(vSamExp, vLen);
// 		
// 		/// get training set
// 		if (m_trainSelectPerLen.GetNum() == 0) {
// 			// init
// 			m_trainSelectPerLen.SetNum(m_pCorpusTrain->GetMaxLen() + 1);
// 			m_trainSelectPerLen.Fill(NULL);
// 			Array<int> aSeq;
// 			for (int i = 0; i < m_pCorpusTrain->GetNum(); i++) {
// 				m_pCorpusTrain->GetSeq(i, aSeq);
// 				int nLen = aSeq.GetNum();
// 				if (!m_trainSelectPerLen[nLen]) {
// 					m_trainSelectPerLen[nLen] = new trf::RandSeq<int>();
// 				}
// 				m_trainSelectPerLen[nLen]->Add(i);
// 			}
// 			for (int i = 0; i < m_trainSelectPerLen.GetNum(); i++) {
// 				if (m_trainSelectPerLen[i])
// 					m_trainSelectPerLen[i]->Random();
// 			}
// 		}
// 
// 		Array<int> aTrainIdx;
// 		for (int len = 1; len<=m_pModel->GetMaxLen(); len++) {
// 			if (!m_trainSelectPerLen[len]) {
// 				lout_error("Cannot find the len=" << len << " in training corpus");
// 			}
// 			for (int i = 0; i < (int)round(vLen[len] * m_nMiniBatchSample); i++) {
// 				aTrainIdx.Add() = m_trainSelectPerLen[len]->Get();
// 			}
// 		}
// 		lout_assert(aTrainIdx.GetNum() == m_nMiniBatchTraining);
// 
// 		// claculate empirical expectation
// 		int nTotalEmpLen = GetEmpiricalExp(vEmpExp, vEmpExp2, aTrainIdx);
// 
// 		lout_assert(nTotalEmpLen == nTotalSampleLen);
// 	}
	double SAfunc::GetSampleLL(CorpusBase *pCorpus, int nCalNum /* = -1 */, int method /* = 0 */)
	{
		int nThread = omp_get_max_threads();

		Array<VocabID> aSeq;
		Vec<double> vSum(nThread);
		Vec<int> vNum(nThread);
		vSum.Fill(0);
		vNum.Fill(0);

		int nCorpusNum = (nCalNum == -1) ? pCorpus->GetNum() : min(nCalNum, pCorpus->GetNum());
		Title::Precent(0, true, nCorpusNum, "GetSampleLL");
#pragma omp parallel for firstprivate(aSeq)
		for (int i = 0; i < nCorpusNum; i++) {
			pCorpus->GetSeq(i, aSeq);

			if (aSeq.GetNum() > m_pModel->GetMaxLen()) {
				continue;
			}

			LogP logprob;
// 			if (method == 0)
// 				logprob = m_pModel->GetLogProbX_AIS(VecShell<VocabID>(aSeq.GetBuffer(), aSeq.GetNum()), m_AISConfigForP.nChain, m_AISConfigForP.nInter);
// 			else
// 				logprob = m_pModel->GetLogProbX_Chib(VecShell<VocabID>(aSeq.GetBuffer(), aSeq.GetNum()), 10);
			logprob = m_pModel->GetLogProb(VecShell<VocabID>(aSeq.GetBuffer(), aSeq.GetNum()));

			vSum[omp_get_thread_num()] += logprob;
			vNum[omp_get_thread_num()]++;
			Title::Precent();
		}

		double dsum = 0;
		int nNum = 0;
		for (int t = 0; t < nThread; t++) {
			dsum += vSum[t];
			nNum += vNum[t];
		}
		return dsum / nNum;
	}
	void SAfunc::IterEnd(double *pFinalParams)
	{
		SetParam(pFinalParams);
		// set the pi as the len-prob in training set.
		m_pModel->SetPi(m_trainPi.GetBuf());
	}
	void SAfunc::WriteModel(int nEpoch)
	{
		String strTempModel;
		String strName = String(m_pathOutputModel).FileName();
#ifdef __linux
		strTempModel.Format("%s.n%d.model", strName.GetBuffer(), nEpoch);
#else
		strTempModel.Format("%s.n%d.model", strName.GetBuffer(), nEpoch);
#endif
		// set the pi as the pi of training set
		m_pModel->SetPi(m_trainPi.GetBuf());
		m_pModel->WriteT(strTempModel);
		m_pModel->SetPi(m_samplePi.GetBuf());
	}
	void SAfunc::GetGradient(double *pdGradient)
	{
		int nWeightNum = m_pModel->GetParamNum();
		m_vEmpExp.Reset(nWeightNum);
		m_vEmpExp2.Reset(nWeightNum);
		m_vSampleExp.Reset(nWeightNum);
		m_vSampleLen.Reset(m_pModel->GetMaxLen() + 1);

#ifdef _CD
		PerfromCD(m_vEmpExp, m_vSampleExp, m_vEmpExp2, m_vSampleLen);
#else

		if (m_bSAMSSample) {
			//GetEmpiricalExp(m_vEmpExp, m_vEmpExp2);
			GetSampleExp(m_vSampleExp, m_vSampleLen);
		}
		else {
			PerfromSA(m_vEmpExp, m_vSampleExp, m_vEmpExp2, m_vSampleLen);
		}
#endif

		/* Calculate the gradient */
		int nFeatNum = m_pModel->m_pFeat->GetNum();
		for (int i = 0; i < nFeatNum; i++) {
			pdGradient[i] = ( m_vEmpFeatExp[i] - m_vSampleExp[i] ) / m_vEmpFeatVar[i];
		}

		for (int i = nFeatNum; i < nWeightNum; i++) {
#ifdef _Var
			double dVar = m_vExp2Value[i - nFeatNum] - pow(m_vExpValue[i - nFeatNum], 2);
			pdGradient[i] = (m_vEmpExp[i] - m_vSampleExp[i]) / max(m_var_gap, dVar);
#else
			pdGradient[i] = m_vEmpExp[i] - m_vSampleExp[i];
#endif
		}

// 		static bool bUpdateVHmat = false;
// 		static int times = 0;
// 		times++;
// 		if (times % 10 == 0) {
// 			bUpdateVHmat = !bUpdateVHmat;
// 		}
// 		if (bUpdateVHmat) {
// 			for (int i = nFeatNum + m_pModel->m_m3dVH.GetSize() + m_pModel->m_m3dCH.GetSize(); i < nWeightNum; i++) {
// 				pdGradient[i] = 0;
// 			}
// 		}
// 		else {
// 			for (int i = nFeatNum; i < nFeatNum + m_pModel->m_m3dVH.GetSize() + m_pModel->m_m3dCH.GetSize(); i++) {
// 				pdGradient[i] = 0;
// 			}
// 		}
		
		

		/*
			Zeta update
		*/
		for (int l = 0; l <= m_pModel->GetMaxLen(); l++) {
			if (m_pModel->m_pi[l] > 0) {
				pdGradient[nWeightNum + l] = m_vSampleLen[l] / m_pModel->m_pi[l];
			}	
			else {
				pdGradient[nWeightNum + l] = 0;
			}
		}

#ifdef _Var
		/* Var update */
		double *pgExp = pdGradient + nWeightNum + GetZetaNum();
		double *pgExp2 = pgExp + m_vExpValue.GetSize();
		for (int i = nFeatNum; i < nWeightNum; i++) {
			pgExp[i - nFeatNum] = m_vEmpExp[i] - m_vExpValue[i - nFeatNum];
			pgExp2[i - nFeatNum] = m_vEmpExp2[i] - m_vExp2Value[i - nFeatNum];
		}
		
		if (m_fvar.Good()) {
			m_fvar.PrintArray("%f ", m_vExpValue.GetBuf(), m_vExpValue.GetSize());
			m_fvar.PrintArray("%f ", m_vExp2Value.GetBuf(), m_vExp2Value.GetSize());
			for (int i = 0; i < m_vExpValue.GetSize(); i++)
				m_fvar.Print("%f ", m_vExp2Value[i] - pow(m_vExpValue[i], 2));
			m_fvar.Print("\n");
			m_fvar.Print("\n");
		}
#endif

		
		if (m_fgrad.Good()) {
			m_fgrad.PrintArray("%f ", pdGradient + m_pModel->m_pFeat->GetNum(), m_pModel->GetParamNum() - m_pModel->m_pFeat->GetNum());
// 			MLfunc::GetGradient(pdGradient);
// 			m_fgrad.PrintArray("%f ", pdGradient + m_pModel->GetParamNum() - GetHHmatSize(), GetHHmatSize());
			m_fgrad.Print("\n");
		}
		if (m_fexp.Good()) {
			m_fexp.PrintArray("%f ", m_vEmpExp.GetBuf() + m_pModel->m_pFeat->GetNum(), m_pModel->GetParamNum() - m_pModel->m_pFeat->GetNum());
			m_fexp.PrintArray("%f ", m_vSampleExp.GetBuf() + m_pModel->m_pFeat->GetNum(), m_pModel->GetParamNum() - m_pModel->m_pFeat->GetNum());
// 			m_fexp.PrintArray("%f ", m_vEmpiricalExp.GetBuf(), nLambdaNum);
// 			m_fexp.PrintArray("%f ", m_vSampleExp.GetBuf(), nLambdaNum);
			m_fexp.Print("\n");
		}
		

		

// 		if (m_vEmpiricalExp[nOffset] == m_vEmpiricalExp[nOffset + 1]) {
// 			//m_pModel->WriteT(m_pathOutputModel);
// 			Pause();
// 		}

// 		VecShell<double> featexp;
// 		MatShell<double> VHexp, HHexp;
// 		m_pModel->BufMap(pdGradient, featexp, VHexp, HHexp);
// 		fileDbg.PrintArray("%f ", featexp.GetBuf(), featexp.GetSize());
// 		fileDbg.PrintArray("%f ", VHexp.GetBuf(), VHexp.GetSize());
// 		fileDbg.PrintArray("%f ", HHexp.GetBuf(), HHexp.GetSize());
// 		fileDbg.Print("\n");
//  		fileDbg.PrintArray("%f ", m_pModel->m_zeta.GetBuf(), m_pModel->GetMaxLen() + 1);
//  		fileDbg.PrintArray("%f ", m_pModel->m_logz.GetBuf(), m_pModel->GetMaxLen() + 1);


	}
	int SAfunc::GetExtraValues(int t, double *pdValues)
	{
		int nValue = 0;

		// set the training pi
		m_pModel->SetPi(m_trainPi.GetBuf());

		Vec<Prob> samsZeta(m_pModel->m_zeta.GetSize());
		Vec<Prob> trueZeta(m_pModel->m_zeta.GetSize());
		//Vec<double> trueLogZ(m_pModel->m_logz.GetSize()); 
		samsZeta.Fill(0);
		trueZeta.Fill(0);
		samsZeta = m_pModel->m_zeta;

		//m_pModel->ApproxNormalize_AIS(m_AISConfigForZ.nChain, m_AISConfigForZ.nInter);

		
		// calcualte LL using exact hidden expectation
		if (m_pCorpusTrain && m_bPrintTrain) pdValues[nValue++] = -GetLL(m_pCorpusTrain);
		if (m_pCorpusValid && m_bPrintValie) pdValues[nValue++] = -GetLL(m_pCorpusValid);
		if (m_pCorpusTest && m_bPrintTest) pdValues[nValue++] = -GetLL(m_pCorpusTest);
			

		/* true Z_L to get the LL */
		if (m_pModel->m_hlayer * m_pModel->m_hnode < 5 && m_pModel->m_pVocab->GetSize() < 100) {
			Vec<LogP> oldZeta(m_pModel->m_zeta.GetSize());
			oldZeta = m_pModel->m_zeta;

			m_pModel->ExactNormalize(); // normalization
			trueZeta.Copy(m_pModel->m_zeta);
			if (m_pCorpusTrain && m_bPrintTrain) pdValues[nValue++] = -GetLL(m_pCorpusTrain);
			if (m_pCorpusValid && m_bPrintValie) pdValues[nValue++] = -GetLL(m_pCorpusValid);
			if (m_pCorpusTest && m_bPrintTest) pdValues[nValue++] = -GetLL(m_pCorpusTest);

			m_pModel->SetZeta(oldZeta.GetBuf());
		}


		/* output debug */
		if (!m_fdbg.Good()) {
			m_fdbg.Open("SAfunc.dbg", "wt");
		}
		m_vAllSampleLenCount *= 1.0 / m_nTotalSample;
		m_vCurSampleLenCount *= 1.0 / m_nMiniBatchSample;
		m_fdbg.Print("pi_cur_: "); m_fdbg.PrintArray("%f ", m_vCurSampleLenCount.GetBuf() + 1, m_vCurSampleLenCount.GetSize() - 1);
		m_fdbg.Print("pi_all_: "); m_fdbg.PrintArray("%f ", m_vAllSampleLenCount.GetBuf() + 1, m_vAllSampleLenCount.GetSize() - 1);
		m_fdbg.Print("pi_true: "); m_fdbg.PrintArray("%f ", m_samplePi.GetBuf() + 1, m_samplePi.GetSize() - 1);
		m_fdbg.Print("z_ais__: "); m_fdbg.PrintArray("%f ", m_pModel->m_zeta.GetBuf() + 1, m_pModel->m_zeta.GetSize() - 1);
		m_fdbg.Print("z_sams_: "); m_fdbg.PrintArray("%f ", samsZeta.GetBuf() + 1, samsZeta.GetSize() - 1);
		m_fdbg.Print("z_true_: "); m_fdbg.PrintArray("%f ", trueZeta.GetBuf() + 1, trueZeta.GetSize() - 1);
		m_fdbg.Print("\n");
		m_vAllSampleLenCount *= m_nTotalSample;
		m_vCurSampleLenCount *= m_nMiniBatchSample;

		m_pModel->SetPi(m_samplePi.GetBuf());

		return nValue;
	}

	void LearningRate::Reset(const char *pstr, int p_t0)
	{
		sscanf(pstr, "%lf,%lf", &tc, &beta);
		t0 = p_t0;
		//lout << "[Learning Rate] tc=" << tc << " beta=" << beta << " t0=" << t0 << endl;
	}
	double LearningRate::Get(int t)
	{
		double gamma;
		if (t <= t0) {
			gamma = 1.0 / (tc + pow(t, beta));
		}
		else {
			gamma = 1.0 / (tc + pow(t0, beta) + t - t0);
		}
		return gamma;
	}


	bool SAtrain::Run(const double *pInitParams /* = NULL */)
	{
		if (!m_pfunc) {
			lout_Solve << "m_pFunc == NULL" << endl;
			return false;
		}
		Clock ck;
		m_dSpendMinute = 0;
		lout.bOutputCmd() = false;

		SAfunc *pSA = (SAfunc*)m_pfunc;
		int nIterPerEpoch = pSA->m_pCorpusTrain->GetNum() / pSA->m_nMiniBatchTraining + 1;
		lout_variable(nIterPerEpoch);

		double *pdCurParams = new double[m_pfunc->GetParamNum()]; //current parameters x_k
		double *pdCurGradient = new double[m_pfunc->GetParamNum()]; //current gradient df_k
		double *pdCurDir = new double[m_pfunc->GetParamNum()]; // current update direction
 		double dCurValue = 0; // function value f_k
		double dExValues[Func::cn_exvalue_max_num]; // save the extra values
		double nExValueNum; // number of extra values

		// if average
		bool bAvg = (m_nAvgBeg > 0);
		double *pdAvgParams = NULL;
		if (bAvg) {
			pdAvgParams = new double[m_pfunc->GetParamNum()];
		}

		//init
		for (int i = 0; i < m_pfunc->GetParamNum(); i++) {
			pdCurParams[i] = (pInitParams) ? pInitParams[i] : 1;
		}
		memset(pdCurGradient, 0, sizeof(double)*m_pfunc->GetParamNum());
		memset(pdCurDir, 0, sizeof(double)*m_pfunc->GetParamNum());

		IterInit(); ///init
		m_pfunc->SetParam(pdCurParams);
		pSA->WriteModel(0);

		// iteration begin
		lout_Solve << "************* Training Begin *****************" << endl;
		lout_Solve << "print-per-iter=" << m_nPrintPerIter << endl;
		lout.bOutputCmd() = false;
		ck.Begin();
		for (m_nIterNum = m_nIterMin; m_nIterNum <= m_nIterMax; m_nIterNum++)
		{
			// epoch number
			m_fEpochNum = 1.0 * m_nIterNum * pSA->m_nMiniBatchSample / pSA->m_pCorpusTrain->GetNum();

			// set the parameter
			m_pfunc->SetParam(pdCurParams);
			// get the gradient
			m_pfunc->GetGradient(pdCurGradient);
			// get the function value
			dCurValue = m_pfunc->GetValue();
			// get the averaged parameters
			if (bAvg) {
				if (m_nIterNum <= m_nAvgBeg) {
					memcpy(pdAvgParams, pdCurParams, sizeof(pdCurParams[0])*m_pfunc->GetParamNum());
				}
				else {
					for (int i = 0; i < m_pfunc->GetParamNum(); i++) {
						pdAvgParams[i] += (pdCurParams[i] - pdAvgParams[i]) / (m_nIterNum - m_nAvgBeg);
					}
				}
			}
			
			
			/* output the values */
			if (m_nIterNum % m_nPrintPerIter == 0 || m_nIterNum == m_nIterMax)
			{
				m_dSpendMinute = ck.ToSecond(ck.Get()) / 60;
				bool bPrintCmd;

				bPrintCmd = lout.bOutputCmd();
				lout.bOutputCmd() = true;
				lout_Solve << "t=" << m_nIterNum;
				cout << setprecision(4) << setiosflags(ios::fixed);
				lout << " epoch=" << m_fEpochNum;
				cout << setprecision(2) << setiosflags(ios::fixed);
				lout << " time=" << m_dSpendMinute << "m";
				lout << (bAvg ? " [Avg]" : " ");
				lout.bOutputCmd() = bPrintCmd;


				// get the ex-value
				if (bAvg) pSA->SetParam(pdAvgParams); ///< set average
				// This function will use AIS to normaliza the model
				nExValueNum = pSA->GetExtraValues(m_nIterNum, dExValues);

				bPrintCmd = lout.bOutputCmd();
				lout.bOutputCmd() = true;
				lout << "ExValues={ ";
				cout << setprecision(3) << setiosflags(ios::fixed);
				for (int i = 0; i < nExValueNum; i++)
					lout << dExValues[i] << " ";
				lout << "}" << endl;

				// write model
				if (m_aWriteAtIter.Find(m_nIterNum) != -1)
					pSA->WriteModel(m_nIterNum);
				// revise the zeta
				for (int i = 1; i < pSA->GetZetaNum(); i++) {
					pdCurParams[i + pSA->GetWeightNum()] = pSA->m_pModel->m_zeta[i];
					pdCurGradient[i + pSA->GetWeightNum()] = 0;
				}

				if (bAvg) pSA->SetParam(pdCurParams); ///< set back
				
				lout.bOutputCmd() = bPrintCmd;
			}

			/* Stop Decision */
			if (StopDecision(m_nIterNum, dCurValue, pdCurGradient)) {
				break;
			}


			// update the learning rate gamma
			UpdateGamma(m_nIterNum);

			// update the direction
			UpdateDir(pdCurDir, pdCurGradient, pdCurParams);

			// Update parameters
			Update(pdCurParams, pdCurDir, 0);

			// Add the spend times
			m_dSpendMinute += ck.ToSecond(ck.End()) / 60;
		}


		lout_Solve << "======== iter:" << m_nIterNum << " ===(" << m_dSpendMinute << "m)=======" << endl;
		lout_Solve << "Iter Finished!" << endl;

		// do something at the end of the iteration
		if (bAvg) pSA->IterEnd(pdAvgParams);
		else pSA->IterEnd(pdCurParams);

		SAFE_DELETE_ARRAY(pdCurGradient);
		SAFE_DELETE_ARRAY(pdCurDir);
		SAFE_DELETE_ARRAY(pdCurParams);
		SAFE_DELETE_ARRAY(pdAvgParams);
		return true;
	}

	void SAtrain::UpdateGamma(int nIterNum)
	{
		m_gamma_lambda = m_gain_lambda.Get(nIterNum);
		m_gamma_hidden = m_gain_hidden.Get(nIterNum);
		m_gamma_zeta = m_gain_zeta.Get(nIterNum);

// 		if (m_fMomentum > 0 && nIterNum > m_gain_lambda.t0) {
// 			m_fMomentum = 0.99;
// 		}

#ifdef _Var
		m_gamma_var = m_gain_var.Get(nIterNum);
		lout_Solve << "g_var=" << m_gamma_var<<endl;
#endif
		
		lout_Solve << "g_lambda=" << m_gamma_lambda
			<< " g_hidden=" << m_gamma_hidden
			<< " g_zeta=" << m_gamma_zeta
			<< " momentum=" << m_fMomentum
			<< endl;
	}
	void SAtrain::UpdateDir(double *pDir, double *pGradient, const double *pdParam)
	{
		/* using the momentum */
		// pdDir is actually the gradient

		SAfunc* pSA = (SAfunc*)m_pfunc;
		int nNgramFeatNum = pSA->GetNgramFeatNum();
		int nWeightNum = pSA->GetWeightNum();
		int nZetaNum = pSA->GetZetaNum();

		// update lambda
		for (int i = 0; i < nNgramFeatNum; i++) {
			pDir[i] = m_gamma_lambda * pGradient[i];
		}
		for (int i = nNgramFeatNum; i < nWeightNum; i++) {
			pDir[i] = m_fMomentum * pDir[i] + m_gamma_hidden * pGradient[i];
		}

		
		
#ifdef _Var
		/* update exp and exp2 */
		for (int i = nWeightNum + nZetaNum; i < pSA->GetParamNum(); i++) {
			pDir[i] = m_gamma_var * pGradient[i];
		}

#endif


		// update zeta
		for (int i = nWeightNum; i < nWeightNum + nZetaNum; i++) {
			pDir[i] = m_gamma_zeta * pGradient[i];
			//pDir[i] = min(m_gamma_zeta, 1.0*pSA->m_pModel->GetMaxLen()*pSA->m_pModel->m_pi[i - nWeightNum]) * pGradient[i];
		}


		// for gap
		int n_dgap_cutnum = CutValue(pDir, nWeightNum, m_dir_gap);
		int n_zgap_cutnum = CutValue(pDir+nWeightNum, nZetaNum, m_zeta_gap);
		lout << "cut-dir="; 
		lout_variable_rate(n_dgap_cutnum, nWeightNum);
		lout << "  cut-zeta=";
		lout_variable_rate(n_dgap_cutnum, nWeightNum);
		lout << endl;
	}
	void SAtrain::Update(double *pdParam, const double *pdDir, double dStep)
	{
		// pdDir is actually the gradient

		SAfunc* pSA = (SAfunc*)m_pfunc;
		int nWeightNum = pSA->GetWeightNum();
		int nZetaNum = pSA->GetZetaNum();

//		lout_assert(nWeightNum == nNgramFeatNum + nVHsize + nCHsize + nHHsize);

		// update lambda
		if (m_bUpdate_lambda) {
			for (int i = 0; i < nWeightNum; i++) {
				pdParam[i] += pdDir[i];
			}

			/* using Nesterov’s Accelerated Gradient momentum setting*/
			/* See "On the importance of initialization and momentum in deep learning" */
			if (m_fMomentum) {
				for (int i=0; i<nWeightNum; i++) {
					pdParam[i] += m_fMomentum * pdDir[i];
				}
			}

#ifdef _Var
			/* update var */
			for (int i = nWeightNum + nZetaNum; i < pSA->GetParamNum(); i++) {
				pdParam[i] += pdDir[i];
			}
#endif

		}

		

		// update zeta
		if (m_bUpdate_zeta) {
			for (int i = nWeightNum; i < nWeightNum + nZetaNum; i++) {
				pdParam[i] += pdDir[i];
			}
			double zeta1 = pdParam[nWeightNum + 1];
			for (int i = nWeightNum + 1; i < nWeightNum + nZetaNum; i++) {
				pdParam[i] -= zeta1; // minus the zeta[1];
			}
		}

		
	}

#define GAIN_INFO(g) lout<<"  "#g"\ttc="<<g.tc<<" beta="<<g.beta<<" t0="<<g.t0<<endl;
	void SAtrain::PrintInfo()
	{
		lout << "[SATrain] *** Info: ***" << endl;
		GAIN_INFO(m_gain_lambda);
		GAIN_INFO(m_gain_hidden);
#ifdef _Var
		GAIN_INFO(m_gain_var);
#endif
		GAIN_INFO(m_gain_zeta);
		lout << "  "; lout_variable(m_dir_gap);
		lout << "  "; lout_variable(m_zeta_gap);
		lout << "[SATrain] *** [End] ***" << endl;
	}

	int SAtrain::CutValue(double *p, int num, double gap)
	{
		int nCutNum = 0;
		if (gap <= 0)
			return nCutNum;

		for (int i = 0; i < num; i++) {
			if (p[i] > gap) {
				p[i] = gap;
				nCutNum++;
			}
			else if (p[i] < -gap) {
				p[i] = -gap;
				nCutNum++;
			}
		}
		return nCutNum;
	}
}