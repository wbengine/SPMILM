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


#include "trf-model.h"

namespace trf
{
	LogP AlgNode::ClusterSum(int *pSeq, int nLen, int nPos, int nOrder)
	{
		m_seq.Set(pSeq, nLen, m_pModel->m_pVocab);
		return m_pModel->ClusterSum(m_seq, nPos, nOrder);
	}

	void Model::Reset(Vocab *pv, int maxlen)
	{
		m_pVocab = pv;
		m_maxlen = maxlen;	
		m_maxSampleLen = (int)(1.02 * maxlen);

		if (maxlen <= 0)
			return;
	

//		SAFE_DELETE(m_pFeat);

		m_pi.Reset(m_maxlen + 1);
		m_logz.Reset(m_maxlen + 1);
		m_zeta.Reset(m_maxlen + 1);
		m_pi.Fill(1);
		m_logz.Fill(0);
		m_zeta.Fill(0);

		// length jump probability
		m_matLenJump.Reset(m_maxSampleLen + 1, m_maxSampleLen + 1);
		m_matLenJump.Fill(0);
		for (int i = 1; i < m_matLenJump.GetRow(); i++) {
			for (int j = max(1, i - 1); j <= min(m_matLenJump.GetCol()-1, i + 1); j++) {
				m_matLenJump[i][j] = 1;
			}
			//m_matLenJump[i][i] = 0; // avoid the self-jump.
			LineNormalize(m_matLenJump[i].GetBuf(), m_matLenJump.GetCol());
		}
	}
	void Model::SetParam(PValue *pValue)
	{
		if (pValue) {
			memcpy(m_value.GetBuf(), pValue, sizeof(pValue[0])*GetParamNum());
		}
	}
	void Model::GetParam(PValue *pValue)
	{
		if (pValue) {
			memcpy(pValue, m_value.GetBuf(), sizeof(pValue[0])*GetParamNum());
		}
	}
	void Model::SetPi(Prob *pPi)
	{
		m_pi.Copy(VecShell<Prob>(pPi, m_pi.GetSize()));
	}
	LogP Model::GetLogProb(Seq &seq, bool bNorm /* = true */)
	{
		Array<int> afeat;
		m_pFeat->Find(afeat, seq);

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
	void Model::LoadFromCorpus(const char *pcorpus, const char *pfeatstyle, int nOrder)
	{
		if (pcorpus) {
			m_pFeat = new Feat(nOrder, m_pVocab->GetClassNum() > 0);
			if (pfeatstyle)
				m_pFeat->Reset(pfeatstyle);
			m_pFeat->LoadFeatFromCorpus(pcorpus, m_pVocab);
			m_value.Reset(m_pFeat->GetNum());
			m_value.Fill(0);
		}
	}
	void Model::FeatCount(Seq &seq, double *pCount, double dadd /* = 1.0 */)
	{
		Array<int> afeat;
		m_pFeat->Find(afeat, seq);
		for (int i = 0; i < afeat.GetNum(); i++) {
			pCount[afeat[i]] += dadd;
		}
	}
	void Model::ReadT(const char *pfilename)
	{
		File fout(pfilename, "rt");
		
		lout << "[Model]: Read(txt) from " << pfilename << endl;

		int nVocabSize = 0;
		fout.Scanf("m_vocabsize=%d\n", &nVocabSize);
		fout.Scanf("m_maxlen=%d\n", &m_maxlen);

		// Reset
		Reset(m_pVocab, m_maxlen);
		if (m_pVocab->GetSize() != nVocabSize) {
			lout_error("[Model] ReadT: the input nVocabSize(" << nVocabSize << ") != m_pVocab->GetSize(" << m_pVocab->GetSize() << ")");
		}

		double dValue;
		fout.Scanf("m_pi=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Scanf("%lf ", &dValue);
			m_pi[i] = dValue;
		}
		fout.Scanf("]\n");
		fout.Scanf("m_logz=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Scanf("%lf ", &dValue);
			m_logz[i] = dValue;
		}
		fout.Scanf("]\n");
		fout.Scanf("m_zeta=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Scanf("%lf ", &dValue);
			m_zeta[i] = dValue;
		}
		fout.Scanf("]\n");

		int nValue = 0;
		fout.Scanf("featnum=%d\n", &nValue);
		m_value.Reset(nValue);
		SAFE_DELETE(m_pFeat);
		m_pFeat = new Feat;
		m_pFeat->m_nTotalNum = nValue;
		m_pFeat->ReadT(fout, m_value.GetBuf());
	}
	void Model::WriteT(const char *pfilename)
	{
		File fout(pfilename, "wt");
		lout << "[Model] Write(txt) to " << pfilename << endl;

		fout.Print("m_vocabsize=%d\n", m_pVocab->GetSize());
		fout.Print("m_maxlen=%d\n", m_maxlen);
		fout.Print("m_pi=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Print("%f ", m_pi[i]);
		}
		fout.Print("]\n");
		fout.Print("m_logz=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Print("%f ", m_logz[i]);
		}
		fout.Print("]\n");
		fout.Print("m_zeta=[ ");
		for (int i = 1; i <= m_maxlen; i++) {
			fout.Print("%f ", m_zeta[i]);
		}
		fout.Print("]\n");

		fout.Print("featnum=%d\n", m_pFeat->GetNum());
		m_pFeat->WriteT(fout, m_value.GetBuf());
	}

	LogP Model::ClusterSum(Seq &seq, int nPos, int nOrder)
	{
		LogP LogSum = 0;
		Array<int> afeat;

		int nLen = seq.GetLen();
		// input nOrder can be larger than the max-order of features.
		int nWordFeatOrder = min(nOrder, GetMaxOrder());

		for (int n = 1; n <= nWordFeatOrder; n++) {
			m_pFeat->Find(afeat, seq, nPos, n);
		}

		// the last cluster
		if (nPos == nLen - nOrder) {
			for (int i = nPos + 1; i < nLen; i++) {
				nWordFeatOrder = min(nLen - i, GetMaxOrder());
				for (int n = 1; n <= nWordFeatOrder; n++) {
					m_pFeat->Find(afeat, seq, i, n);
				}
			}
		}

		for (int i = 0; i < afeat.GetNum(); i++)
			LogSum += m_value[afeat[i]];

		return LogSum;
	}
	double Model::ExactNormalize(int nLen)
	{
		int nMaxOrder = GetMaxOrder();
		LogP logZ = LogP_zero;

		/* If the length is less than order, then we enumerate all the sequence of such length */
		if (nLen <= nMaxOrder) {
			Seq seq(nLen);
			vIter<VocabID> SeqIter(seq.GetWordSeq(), nLen);
			SeqIter.AddAllLine(0, m_pVocab->GetSize() - 1);
			while (SeqIter.Next()) {
				seq.SetClass(m_pVocab);
				double d = GetLogProb(seq, false);
				logZ = Log_Sum(logZ, d);
			}
		}
		else {
			m_AlgNode.ForwardBackward(nLen, nMaxOrder, m_pVocab->GetSize());
			logZ = m_AlgNode.GetLogSummation();
		}

		m_logz[nLen] = logZ;
		return logZ;
	}
	void Model::ExactNormalize()
	{
		for (int len = 1; len <= m_maxlen; len++) {
			ExactNormalize(len);
			m_zeta[len] = m_logz[len] - m_logz[1];
			//lout << " logZ[" << len << "] = " << m_logz[len] << endl;
		}
	}
	void Model::GetNodeExp(int nLen, double *pExp)
	{
		memset(pExp, 0, sizeof(pExp[0])*GetParamNum());

		int nMaxOrder = GetMaxOrder();
		/* If the length is less than order, then we enumerate all the sequence of such length */
		if (nLen <= nMaxOrder) {
			Seq seq(nLen);
			vIter<VocabID> SeqIter(seq.GetWordSeq(), nLen);
			SeqIter.AddAllLine(0, m_pVocab->GetSize() - 1);
			while (SeqIter.Next()) {
				seq.SetClass(m_pVocab);
				Prob prob = LogP2Prob(GetLogProb(seq));
				Array<int> afeat;
				m_pFeat->Find(afeat, seq);
				for (int i = 0; i < afeat.GetNum(); i++) {
					pExp[afeat[i]] += prob;
				}
			}
		}
		else {
			int nClusterNum = nLen - nMaxOrder + 1;
			// circle for the position pos
			for (int pos = 0; pos < nClusterNum; pos++) {
				// ergodic the cluster
				Seq seq(nLen);
				vIter<VocabID> SeqIter(seq.GetWordSeq() + pos, nMaxOrder);
				SeqIter.AddAllLine(0, m_pVocab->GetSize() - 1);
				while (SeqIter.Next()) {
					seq.SetClass(m_pVocab);
					Prob prob = LogP2Prob(m_AlgNode.GetMarginalLogProb(pos, seq.GetWordSeq() + pos, nMaxOrder, m_logz[nLen]));
					Array<int> afeat;
					for (int n = 1; n <= nMaxOrder; n++)
						m_pFeat->Find(afeat, seq, pos, n);
					for (int i = 0; i < afeat.GetNum(); i++) {
						pExp[afeat[i]] += prob;
					}

					//////////////////////////////////////////////////////////////////////////
					// the last cluster
					//////////////////////////////////////////////////////////////////////////
					if (pos == nClusterNum - 1) {
						afeat.Clean();
						for (int ii = 1; ii < nMaxOrder; ii++) { // position ii
							for (int n = 1; n <= nMaxOrder - ii; n++) { // order n
								m_pFeat->Find(afeat, seq, pos + ii, n);
							}
						}
						for (int i = 0; i < afeat.GetNum(); i++) {
							pExp[afeat[i]] += prob;
						}
					}
				}
			}
		}
	}
	void Model::GetNodeExp(double *pExp, Prob *pLenProb/* = NULL*/)
	{
		if (pLenProb == NULL)
			pLenProb = m_pi.GetBuf();
		VecShell<double> exp(pExp, GetParamNum());
		Vec<double> expTemp(GetParamNum());

		exp.Fill(0);
		for (int len = 1; len <= m_maxlen; len++) {

			int nMaxOrder = GetMaxOrder(); ///< max-order
			m_AlgNode.ForwardBackward(len, nMaxOrder, m_pVocab->GetSize());

			GetNodeExp(len, expTemp.GetBuf());

			for (int i = 0; i < exp.GetSize(); i++) {
				exp[i] += pLenProb[len] * expTemp[i];
			}
		}
	}
	
	void Model::Sample(Seq &seq)
	{
		LocalJump(seq);
		MarkovMove(seq);
	}
	void Model::LocalJump(Seq &seq)
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
			lout_error("[Model] Sample: nNewLen(" << nNewLen << ") and nOldLen(" << nOldLen << ")");
		}


		if (Acceptable(LogP2Prob(logpAcc))) {
			seq.Reset(nNewLen);
			m_nLenJumpAccTimes++;
		}
		else {
			seq.Reset(nOldLen);
		}
		m_nLenJumpTotalTime++;

	}
	void Model::MarkovMove(Seq &seq)
	{
		/* Gibbs sampling */
		for (int nPos = 0; nPos < seq.GetLen(); nPos++) {
			SampleC(seq, nPos);
			SampleX(seq, nPos);
		}
	}

	LogP Model::ProposeLength(int nOld, int &nNew, bool bSample)
	{
		if (bSample) {
			nNew = LineSampling(m_matLenJump[nOld].GetBuf(), m_matLenJump[nOld].GetSize());
		}

		return Prob2LogP(m_matLenJump[nOld][nNew]);
	}
	LogP Model::ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample)
	{
		/* if there are no class, then return 0 */
		if (m_pVocab->GetClassNum() == 0) {
			ci = VocabID_none;
			return 0;
		}

		Vec<LogP> vlogps(m_pVocab->GetClassNum());
		ProposeCProbs(vlogps, seq, nPos);

		if (bSample) {
			ci = LogLineSampling(vlogps.GetBuf(), vlogps.GetSize());
		}

		return vlogps[ci];
	}
	void Model::ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos)
	{
// 		logps.Fill(Prob2LogP(1.0 /m_pVocab->GetClassNum()));
// 		return;

		VocabID savecid = seq.x[class_layer][nPos];
		for (int cid = 0; cid < m_pVocab->GetClassNum(); cid++) {
			seq.x[class_layer][nPos] = cid;
			logps[cid] = GetReducedModelForC(seq, nPos);
		}
		seq.x[class_layer][nPos] = savecid;
		LogLineNormalize(logps.GetBuf(), m_pVocab->GetClassNum());
	}
	LogP Model::GetReducedModelForC(Seq &seq, int nPos)
	{
		if (seq.x[class_layer][nPos] == VocabID_none)
			return 0;

		LogP logSum = 0;
		// class ngram features
		Array<int> afeat;
		m_pFeat->FindPosDep(afeat, seq, nPos, 1);
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	LogP Model::GetReducedModelForW(Seq &seq, int nPos)
	{
		LogP logSum = 0;
		Array<int> afeat;
		m_pFeat->FindPosDep(afeat, seq, nPos, 2); ///< containing word
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	LogP Model::GetReducedModel(Seq &seq, int nPos)
	{
		LogP logSum = 0;
		Array<int> afeat;
		m_pFeat->FindPosDep(afeat, seq, nPos, 0); // all
		for (int i = 0; i < afeat.GetNum(); i++) {
			logSum += m_value[afeat[i]];
		}

		return logSum;
	}
	LogP Model::GetMarginalProbOfC(Seq &seq, int nPos)
	{
		LogP resLogp = LogP_zero;

		Array<VocabID> *pXs = m_pVocab->GetWord(seq.x[class_layer][nPos]);

		VocabID saveX = seq.x[word_layer][nPos];
		for (int i = 0; i < pXs->GetNum(); i++) {
			seq.x[word_layer][nPos] = pXs->Get(i);
			/* Only need to calculate the summation of weight depending on x[nPos], c[nPos] */
			/* used to sample the c_i */
			resLogp = Log_Sum(resLogp, GetReducedModel(seq, nPos));
			//resLogp = Log_Sum(resLogp, GetLogProb(seq, false));
		}
		seq.x[word_layer][nPos] = saveX;

		return resLogp;
	}
	void Model::SampleC(Seq &seq, int nPos)
	{
		if (m_pVocab->GetClassNum() == 0) {
			seq.x[class_layer][nPos] = VocabID_none;
			return;
		}

		/* Sample C0 */
		Vec<LogP> vlogps_c(m_pVocab->GetClassNum());
		ProposeCProbs(vlogps_c, seq, nPos);
		VocabID ci = seq.x[class_layer][nPos];
		VocabID C0 = LogLineSampling(vlogps_c.GetBuf(), vlogps_c.GetSize());
		LogP logpRi = vlogps_c[ci];
		LogP logpR0 = vlogps_c[C0];


		/* Calculate the probability p_t(h, c) */
		seq.x[class_layer][nPos] = ci;
		LogP Logp_ci = GetMarginalProbOfC(seq, nPos);
		seq.x[class_layer][nPos] = C0;
		LogP Logp_C0 = GetMarginalProbOfC(seq, nPos);

		LogP acclogp = logpRi + Logp_C0 - (logpR0 + Logp_ci);

		m_nSampleHTotalTimes++;
		if (Acceptable(LogP2Prob(acclogp))) {
			m_nSampleHAccTimes++;
			seq.x[class_layer][nPos] = C0;
		}
		else {
			seq.x[class_layer][nPos] = ci;
		}
	}
	LogP Model::SampleX(Seq &seq, int nPos, bool bSample/* = true*/)
	{
		/*
		The function calculate G(x_i| x_{other}, h)
		if bSample is true, draw a sample for x_i;
		otherwise, only calcualte the conditional probability.
		*/
		if (nPos >= seq.GetLen()) {
			lout_error("[Model] SampleH: the nPos(" << nPos << ") > the length of sequence(" << seq.GetLen() << ")");
		}

		Array<VocabID> *pXs = m_pVocab->GetWord(seq.x[class_layer][nPos]);
		Array<LogP> aLogps;

		VocabID nSaveX = seq.x[word_layer][nPos]; // save w[nPos]
		for (int i = 0; i < pXs->GetNum(); i++) {
			seq.x[word_layer][nPos] = pXs->Get(i);
			/* To reduce the computational cost, instead of GetLogProb,
			we just need to calculate the summation of weight depending on x[nPos]
			*/
			aLogps[i] = GetReducedModelForW(seq, nPos);
		}
		LogLineNormalize(aLogps, pXs->GetNum());

		int idx;
		if (bSample) {
			/* sample a value for x[nPos] */
			idx = LogLineSampling(aLogps, pXs->GetNum());
			seq.x[word_layer][nPos] = pXs->Get(idx);
		}
		else {
			idx = pXs->Find(nSaveX); // find nSave in the array.
			seq.x[word_layer][nPos] = nSaveX;
			if (idx == -1) {
				lout_error("Can't find the VocabID(" << nSaveX << ") in the array.\n"
					<< "This may beacuse word(" << nSaveX << ") doesnot belongs to class(" 
					<< seq.x[class_layer][nPos] << ")");
			}
		}

		return aLogps[idx];
	}

	LogP Model::AISNormalize(int nLen, int nChain, int nInter)
	{
		int nParamsNum = GetParamNum();

		Vec<PValue> vParamsPn(nParamsNum);
		Vec<PValue> vParamsP0(nParamsNum);
		Vec<PValue> vParamsCur(nParamsNum);
		this->GetParam(vParamsP0.GetBuf());

		/* set the P_n */
		/* Set with all the unigram values, i.e. all the VH and CH */
		vParamsPn.Fill(0);
		/* calculate the normalization constants of P_n */
		LogP logz_pn = nLen * log((double)m_pVocab->GetSize());


		/* In the intermediate models,
		these models share the word/class ngram features,
		to save the memory cost.
		*/
		Model *pInterModel = new Model(m_pVocab, m_maxlen);
		pInterModel->m_pFeat = m_pFeat;
		pInterModel->m_value.Reset(GetParamNum());
		
		// Weight for each chain
		Array<LogP> aLogWeight; 
		aLogWeight.SetNum(nChain);

		LogP localLogSum = LogP_zero; // save the current results
		int  localChainNum = 0;

		for (int k = 0; k < nChain; k++) {
			PValue* pParamsCur = vParamsCur.GetBuf();
			PValue *pP0 = vParamsP0.GetBuf();
			PValue *pPn = vParamsPn.GetBuf();

			Seq seq(nLen);
			/* As here we set the P_n is the uniform distribution. */
			seq.Random(m_pVocab); 
			pInterModel->SetParam(vParamsPn.GetBuf());
			//LogP logp_old = -(log(2) * GetHNode() + log(m_pVocab->GetSize())) * nLen;
			LogP logp_old = pInterModel->GetLogProb(seq, false) - logz_pn;

			double log_w = 0;
			for (int t = nInter - 1; t >= 0; t--) {
				/* set the intermediate parameters */
				double beta = GetAISFactor(t, nInter);
				for (int i = 0; i < nParamsNum; i++)
					pParamsCur[i] = pP0[i] * (1 - beta) + pPn[i] * beta;
				pInterModel->SetParam(pParamsCur);

				/* compute the weight */
				LogP rate = pInterModel->GetLogProb(seq, false) - logp_old;
				log_w += rate;


				/* sample sequence*/
				pInterModel->MarkovMove(seq);
				logp_old = pInterModel->GetLogProb(seq, false);
			}

			aLogWeight[k] = log_w; // record the log-weight


			
			localLogSum = Log_Sum(localLogSum, log_w);
			localChainNum = localChainNum + 1;
			LogP localLogz = localLogSum - log(localChainNum);
			lout << localLogz << "(" << localChainNum << ") ";
		}

		
		pInterModel->m_pFeat = NULL; // avoid to release the feature buffer
		SAFE_DELETE(pInterModel);
		

		LogP logz = Log_Sum(aLogWeight.GetBuffer(), aLogWeight.GetNum()) - Prob2LogP(nChain);

		m_logz[nLen] = logz;
		return logz;
	}
	void Model::AISNormalize(int nChain, int nInter)
	{
		int nParamsNum = GetParamNum();

		Vec<PValue> vParamsPn(nParamsNum);
		Vec<PValue> vParamsP0(nParamsNum);
		Vec<PValue> vParamsCur(nParamsNum);
		this->GetParam(vParamsP0.GetBuf());

		/* set the P_n */
		/* Set with all the unigram values, i.e. all the VH and CH */
		vParamsPn.Fill(0);
		/* calculate the normalization constants of P_n for each length */
		Vec<LogP> alogz_pn(m_maxlen + 1);
		for (int i = 0; i <= m_maxlen; i++)
			alogz_pn[i] = i * log((double)m_pVocab->GetSize());


		/* In the intermediate models,
		these models share the features,
		to save the memory cost.
		*/
		Model *pInterModel = new Model(m_pVocab, m_maxlen);
		pInterModel->m_pFeat = m_pFeat;
		pInterModel->m_value.Reset(GetParamNum());
		pInterModel->SetParam(vParamsPn.GetBuf());

		// Weight for length, for each chain
		Mat<LogP> matLogWeight(m_maxlen+1, nChain);
		Mat<LogP> matLogPOld(m_maxlen+1, nChain);
		Mat<Seq*> matSeq(m_maxlen+1, nChain);
		matLogWeight.Fill(0);
		matLogPOld.Fill(0);
		matSeq.Fill(0);
		for (int i = 1; i <= m_maxlen; i++) {
			for (int j = 0; j < nChain; j++) {
				matSeq[i][j] = new Seq(i);
				matSeq[i][j]->Random(m_pVocab);
				matLogPOld[i][j] = pInterModel->GetLogProb(*matSeq[i][j], false) - alogz_pn[i];
			}
		}
		

		// for each intermediate distribution
		lout.Progress(0, true, nInter, "AIS");
		for (int t = nInter - 1; t >= 0; t--) {
			PValue* pParamsCur = vParamsCur.GetBuf();
			PValue *pP0 = vParamsP0.GetBuf();
			PValue *pPn = vParamsPn.GetBuf();

			// get the intermediate parameters
			double beta = GetAISFactor(t, nInter);
			for (int i = 0; i < nParamsNum; i++)
				pParamsCur[i] = pP0[i] * (1 - beta) + pPn[i] * beta;
			pInterModel->SetParam(pParamsCur);

#pragma omp parallel for
			for (int nLen = 1; nLen <= m_maxlen; nLen++) { // for each length
				for (int k = 0; k < nChain; k++) { // for each chain
					/* compute the weight */
					LogP rate = pInterModel->GetLogProb(*matSeq[nLen][k], false) - matLogPOld[nLen][k];
					matLogWeight[nLen][k] += rate;


					/* sample sequence*/
					pInterModel->MarkovMove(*matSeq[nLen][k]);
					matLogPOld[nLen][k] = pInterModel->GetLogProb(*matSeq[nLen][k], false);
				}
			}

			lout.Progress(nInter-t);
		}

		pInterModel->m_pFeat = NULL; // avoid to release the feature buffer
		SAFE_DELETE(pInterModel);
		for (int i = 1; i <= m_maxlen; i++) {
			for (int j = 0; j < nChain; j++) {
				SAFE_DELETE(matSeq[i][j]);
			}
		}

		for (int nLen = 1; nLen <= m_maxlen; nLen++) {
			LogP logz = Log_Sum(matLogWeight[nLen].GetBuf(), matLogWeight[nLen].GetSize()) - Prob2LogP(nChain);
			m_logz[nLen] = logz;
			lout << "logz[" << nLen << "] = " << logz << endl;
		}
	}
}