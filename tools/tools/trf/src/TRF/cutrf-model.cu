#include "cutrf-model.h"
#include <device_launch_parameters.h>

namespace cutrf
{
	__host__ void Model::Copy(trf::Model &m, int nMaxThread) {
		m_feat.Copy(*m.m_pFeat, nMaxThread);
		m_value.Copy(m.m_value);
		m_maxlen = m.m_maxlen;
		m_maxOrder = m.GetMaxOrder();
		m_pi.Copy(m.m_pi);
		m_logz.Copy(m.m_logz);
		m_zeta.Copy(m.m_zeta);
		m_vocab.Copy(*m.m_pVocab);
		m_matLenJump.Copy(m.m_matLenJump);
		m_maxSampleLen = m.m_maxSampleLen;

		CUDA_CALL(cudaMalloc(&m_pFindBuf, sizeof(int) * nMaxThread * FEAT_FIND_MAX));
		m_matClassSampleBuf.Reset(nMaxThread, m.m_pVocab->GetClassNum());
		m_matWordSampleBuf.Reset(nMaxThread, m_vocab.GetMaxWordInOneClass());

		m_vRandState.Reset(nMaxThread);
		Kernal_InitState << <ceil(1.0*nMaxThread / 128), 128 >> >(m_vRandState.GetBuf(), nMaxThread);
	}

	__device__ LogP Model::GetLogProb(Seq &seq, bool bNorm /*= true*/)
	{
		cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
		m_feat.Find(afeat, seq);

		LogP logSum = 0;
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		if (bNorm) {
			int nLen = min(m_maxlen, seq.GetLen());
			logSum = logSum - m_logz[nLen] + Prob2LogP(m_pi[nLen]);
		}
		return logSum;
	}
	__device__ void Model::Sample(Seq &seq)
	{
		LocalJump(seq);
		MarkovMove(seq);
	}
	__device__ void Model::LocalJump(Seq &seq)
	{
		int nOldLen = seq.GetLen();
		int nNewLen = 0;
		LogP j1 = ProposeLength(nOldLen, nNewLen, true);
		LogP j2 = ProposeLength(nNewLen, nOldLen, false);

		if (nNewLen == nOldLen)
			return;

		LogP logpAcc = 0;
		if (nNewLen == nOldLen + 1) {
			LogP logpold = GetLogProb(seq);
			seq.Reset(nNewLen);
			LogP R = ProposeC0(seq.x[class_layer][nNewLen - 1], seq, nNewLen - 1, true);
			LogP G = SampleX(seq, nNewLen - 1);
			LogP logpnew = GetLogProb(seq);

			logpAcc = (j2 - j1) + logpnew - (logpold + R + G);
		}
		else if (nNewLen == nOldLen - 1) {
			LogP logpold = GetLogProb(seq);
			LogP R = ProposeC0(seq.x[class_layer][nOldLen - 1], seq, nOldLen - 1, false);
			LogP G = SampleX(seq, nOldLen - 1, false);

			seq.Reset(nNewLen);
			LogP logpnew = GetLogProb(seq);

			logpAcc = (j2 - j1) + logpnew + R + G - logpold;
		}
		else if (nNewLen != nOldLen){
			printf("[ERROR] [Model] Sample: nNewLen(%d) and nOldLen(%d)", nNewLen, nOldLen);
		}


		if (Acceptable(LogP2Prob(logpAcc), &m_vRandState[_TID])) {
			seq.Reset(nNewLen);
		}
		else {
			seq.Reset(nOldLen);
		}

	}
	/// [sample] Markov Move - perform the gibbs sampling
	__device__ void Model::MarkovMove(Seq &seq)
	{
		/* Gibbs sampling */
		for (int nPos = 0; nPos < seq.GetLen(); nPos++) {
			SampleC(seq, nPos);
			SampleX(seq, nPos);
		}
	}
	/// [sample] Propose the length, using the variable m_matLenJump
	__device__ LogP Model::ProposeLength(int nOld, int &nNew, bool bSample)
	{
		if (bSample) {
			nNew = LineSampling(m_matLenJump[nOld].GetBuf(), m_matLenJump[nOld].GetSize(), &m_vRandState[_TID]);
		}

		return Prob2LogP(m_matLenJump[nOld][nNew]);
	}
	/// [sample] Cuda application for trf::Model::GetReducedModelForC
	__device__ LogP Model::GetReducedModelForC(Seq &seq, int nPos)
	{
		if (seq.x[class_layer][nPos] == VocabID_none)
			return 0;

		LogP logSum = 0;
		int nlen = seq.GetLen();
		int nMaxOrder = m_maxOrder;
		// class ngram features
		cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
		for (int order = 1; order <= nMaxOrder; order++) {
			for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
				m_feat.FindClass(afeat, seq, i, order);
			}
		}
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	/// [sample] Cuda application for trf::Model::GetReducedModelForW
	__device__ LogP Model::GetReducedModelForW(Seq &seq, int nPos)
	{
		LogP logSum = 0;
		int nlen = seq.GetLen();
		int nMaxOrder = m_maxOrder;
		// class ngram features
		cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
		for (int order = 1; order <= nMaxOrder; order++) {
			for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
				m_feat.FindWord(afeat, seq, i, order);
			}
		}
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	/// [sample] A unnormalized reduced depending on nPos.
	__device__ LogP Model::GetReducedModel(Seq &seq, int nPos)
	{
		LogP logSum = 0;
		int nlen = seq.GetLen();
		int nMaxOrder = m_maxOrder;
		// class ngram features
		cu::Array<int> afeat(FEAT_FIND_BUF(m_pFindBuf), FEAT_FIND_MAX);
		for (int order = 1; order <= nMaxOrder; order++) {
			for (int i = max(0, nPos - order + 1); i <= min(nlen - order, nPos); i++) {
				m_feat.Find(afeat, seq, i, order);
			}
		}
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	/// [sample] given c_i, summate the probabilities of x_i, i.e. P(c_i)
	__device__ LogP Model::GetMarginalProbOfC(Seq &seq, int nPos)
	{
		LogP resLogp = LOGP_ZERO;

		cu::VecShell<VocabID> vXs = m_vocab.GetWord(seq.x[class_layer][nPos]);

		VocabID saveX = seq.x[word_layer][nPos];
		for (int i = 0; i < vXs.GetSize(); i++) {
			seq.x[word_layer][nPos] = vXs[i];
			/* Only need to calculate the summation of weight depending on x[nPos], c[nPos] */
			/* used to sample the c_i */
			resLogp = Log_Sum(resLogp, GetReducedModel(seq, nPos));
			//resLogp = Log_Sum(resLogp, GetLogProb(seq, false));
		}
		seq.x[word_layer][nPos] = saveX;

		return resLogp;
	}
	/// [sample] Propose the c_{i} at position i. Then return the propose probability R(c_i|h_i,c_{other})
	__device__ LogP Model::ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample)
	{
		/* if there are no class, then return 0 */
		int nClassNum = m_vocab.GetClassNum();
		if (nClassNum == 0) {
			ci = VocabID_none;
			return 0;
		}

		cu::VecShell<LogP> vlogps = m_matClassSampleBuf[_TID];
		ProposeCProbs(vlogps, seq, nPos);

		if (bSample) {
			ci = LogLineSampling(vlogps.GetBuf(), vlogps.GetSize(), &m_vRandState[_TID]);
		}

		return vlogps[ci];
	}
	/// [sample] Return the propose distribution of c_i at position nPos
	__device__ void Model::ProposeCProbs(cu::VecShell<LogP> logps, Seq &seq, int nPos)
	{
		VocabID savecid = seq.x[class_layer][nPos];
		int nClassNum = m_vocab.GetClassNum();
		for (int cid = 0; cid < nClassNum; cid++) {
			seq.x[class_layer][nPos] = cid;
			logps[cid] = GetReducedModelForC(seq, nPos);
		}
		seq.x[class_layer][nPos] = savecid;
		LogLineNormalize(logps.GetBuf(), nClassNum);
	}
	/// [sample] Sample the c_i at position nPos without x_i.
	__device__ void Model::SampleC(Seq &seq, int nPos)
	{
		int nClassNum = m_vocab.GetClassNum();

		if (m_vocab.GetClassNum() == 0) {
			seq.x[class_layer][nPos] = VocabID_none;
			return;
		}

		/* Sample C0 */
		cu::VecShell<LogP> vlogps_c = m_matClassSampleBuf[_TID];
		ProposeCProbs(vlogps_c, seq, nPos);
		VocabID ci = seq.x[class_layer][nPos];
		VocabID C0 = LogLineSampling(vlogps_c.GetBuf(), vlogps_c.GetSize(), &m_vRandState[_TID]);
		LogP logpRi = vlogps_c[ci];
		LogP logpR0 = vlogps_c[C0];


		/* Calculate the probability p_t(h, c) */
		seq.x[class_layer][nPos] = ci;
		LogP Logp_ci = GetMarginalProbOfC(seq, nPos);
		seq.x[class_layer][nPos] = C0;
		LogP Logp_C0 = GetMarginalProbOfC(seq, nPos);

		LogP acclogp = logpRi + Logp_C0 - (logpR0 + Logp_ci);

		if (Acceptable(LogP2Prob(acclogp), &m_vRandState[_TID])) {
			seq.x[class_layer][nPos] = C0;
		}
		else {
			seq.x[class_layer][nPos] = ci;
		}
	}
	/// [sample] Sample the x_i at position nPos
	/* if bSample=ture, then sample x[nPos]. Otherwise only calculate the conditional probabilities of current x[nPos]. */
	__device__ LogP Model::SampleX(Seq &seq, int nPos, bool bSample /*= true*/)
	{
		/*
		The function calculate G(x_i| x_{other}, h)
		if bSample is true, draw a sample for x_i;
		otherwise, only calcualte the conditional probability.
		*/
		cu::VecShell<VocabID> vXs = m_vocab.GetWord(seq.x[class_layer][nPos]);
		cu::Array<LogP> aLogps(m_matWordSampleBuf[_TID].GetBuf(), m_matWordSampleBuf.GetCol());

		VocabID nSaveX = seq.x[word_layer][nPos]; // save w[nPos]
		for (int i = 0; i < vXs.GetSize(); i++) {
			seq.x[word_layer][nPos] = vXs[i];
			/* To reduce the computational cost, instead of GetLogProb,
			we just need to calculate the summation of weight depending on x[nPos]
			*/
			aLogps[i] = GetReducedModelForW(seq, nPos);
		}
		LogLineNormalize(aLogps.GetBuf(), vXs.GetSize());

		int idx;
		if (bSample) {
			/* sample a value for x[nPos] */
			idx = LogLineSampling(aLogps.GetBuf(), vXs.GetSize(), &m_vRandState[_TID]);
			seq.x[word_layer][nPos] = vXs[idx];
		}
		else {
			// find nSave in the array
			idx = -1;
			for (int i = 0; i < vXs.GetSize(); i++) {
				if (vXs[i] == nSaveX) {
					idx = i;
					break;
				}
			}
			//idx = pXs->Find(nSaveX);
			seq.x[word_layer][nPos] = nSaveX;
			if (idx == -1) {
				printf("Can't find the VocabID(%d) in the the class(%d)\n", nSaveX, seq.x[class_layer][nPos]);
			}
		}

		return aLogps[idx];
	}


	__global__ void Kernal_Sample(
		cu::Array<cutrf::Seq> aSeq,
		cutrf::Model m)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		m.Sample(aSeq[tid]);
	}

	void cudaSample(cu::Array<cutrf::Seq> dev_aSeq, cutrf::Model m)
	{
		Kernal_Sample << <1, dev_aSeq.GetNum() >> >(dev_aSeq, m);
	}
}
