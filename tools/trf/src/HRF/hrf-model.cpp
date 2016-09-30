#include "hrf-model.h"

namespace hrf
{
	void Seq::Reset(int len, int hlayer, int hnode)
	{
		if (m_nLen != len || GetHlayer() != hlayer || GetHnode() != hnode) {
			m_nLen = len;
			m_hlayer = hlayer;
			m_hnode = hnode;
			x.Reset(len);
			h.Reset(len, hlayer * hnode);
		}
	}
	void Seq::Copy(Seq &seq)
	{
		x.Copy(seq.x);
		h.Copy(seq.h);
		m_nLen = seq.m_nLen;
	}
	Seq Seq::GetSubSeq(int nPos, int nOrder)
	{
		if (nPos + nOrder > m_nLen) {
			lout_error("[Seq] GetSubSeq: nPos+nOrder > nLen!!");
		}
		Seq sub(nOrder, GetHlayer(), GetHnode());
		for (int i = nPos; i < nPos + nOrder; i++) {
			sub.x.GetWordSeq()[i - nPos] = x.GetWordSeq()[i];
			sub.x.GetClassSeq()[i - nPos] = x.GetClassSeq()[i];
			sub.h[i - nPos] = h[i];
		}
		return sub;
	}
	bool Seq::operator==(Seq &s)
	{
		if (GetLen() != s.GetLen())
			return false;

		if (x.x == s.x.x && h == s.h)
			return true;
		return false;
	}
	void Seq::Print()
	{
		for (int i = 0; i < h.GetCol(); i++) {
			for (int j = 0; j < m_nLen; j++) {
				lout << h[j][i] << "\t";
			}
			lout << endl;
		}
		x.Print();
	}
	void Seq::Write(File &file)
	{
		ofstream ofile(file.fp);
		for (int i = 0; i < h.GetCol(); i++) {
			for (int j = 0; j < m_nLen; j++) {
				ofile << h[j][i] << "\t";
			}
			ofile << endl;
		}
		x.Print(file);
	}

	void Model::Reset(Vocab *pv, int hlayer, int hnode, int maxlen)
	{
		trf::Model::Reset(pv, maxlen);

		m_hlayer = hlayer;
		m_hnode = hnode;
		m_m3dVH.Reset(m_pVocab->GetSize(), m_hlayer * m_hnode, 2); // 0 and 1
		m_m3dCH.Reset(m_pVocab->GetClassNum(), m_hlayer * m_hnode, 2); // 0 and 1
		m_m3dHH.Reset(m_hlayer * m_hnode, m_hnode, 4); // 0-0, 0-1, 1-0, 1-1
	}
	void Model::SetParam(PValue *pParam)
	{
		trf::Model::SetParam(pParam);
		pParam += m_pFeat->GetNum();
		memcpy(m_m3dVH.GetBuf(), pParam, sizeof(PValue)*m_m3dVH.GetSize());
		pParam += m_m3dVH.GetSize();
		memcpy(m_m3dCH.GetBuf(), pParam, sizeof(PValue)*m_m3dCH.GetSize());
		pParam += m_m3dCH.GetSize();
		memcpy(m_m3dHH.GetBuf(), pParam, sizeof(PValue)*m_m3dHH.GetSize());
	}
	void Model::GetParam(PValue *pParam)
	{
		trf::Model::GetParam(pParam);
		pParam += m_pFeat->GetNum();
		memcpy(pParam, m_m3dVH.GetBuf(), sizeof(PValue)*m_m3dVH.GetSize());
		pParam += m_m3dVH.GetSize();
		memcpy(pParam, m_m3dCH.GetBuf(), sizeof(PValue)*m_m3dCH.GetSize());
		pParam += m_m3dCH.GetSize();
		memcpy(pParam, m_m3dHH.GetBuf(), sizeof(PValue)*m_m3dHH.GetSize());
	}
	LogP Model::GetLogProb(Seq &seq, bool bNorm /* = true */)
	{
		LogP logSum = trf::Model::GetLogProb(seq.x, false);

		// Vocab * Hidden
		for (int i = 0; i < seq.GetLen(); i++) {
			logSum += SumVHWeight(m_m3dVH[seq.wseq()[i]], seq.h[i]);
		}

		// Class * Hidden
		if (m_m3dCH.GetSize() > 0) {
			for (int i = 0; i < seq.GetLen(); i++) {
				logSum += SumVHWeight(m_m3dCH[seq.cseq()[i]], seq.h[i]);
			}
		}

		// Hidden * Hidden
		for (int i = 0; i < seq.GetLen() - 1; i++) {
			logSum += SumHHWeight(m_m3dHH, seq.h[i], seq.h[i + 1]);
		}

		// normalization
		if (bNorm) {
			logSum = logSum - m_logz[seq.GetLen()] + trf::Prob2LogP(m_pi[seq.GetLen()]);
		}
		return logSum;
	}

	void Model::ReadT(const char *pfilename)
	{
		File fout(pfilename, "rt");

		lout << "[Model]: Read(txt) from " << pfilename << endl;

		int nVocabSize = 0;
		fout.Scanf("m_vocabsize=%d\n", &nVocabSize);
		fout.Scanf("m_maxlen=%d\n", &m_maxlen);
		fout.Scanf("m_hlayer=%d\n", &m_hlayer);
		fout.Scanf("m_hnode=%d\n", &m_hnode);
		// Reset
		Reset(m_pVocab, m_hlayer, m_hnode, m_maxlen);
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
		m_pFeat = new trf::Feat;
		m_pFeat->m_nTotalNum = nValue;
		m_pFeat->ReadT(fout, m_value.GetBuf());

		/* Init all the values */
		m_m3dVH.Reset();
		m_m3dCH.Reset();
		m_m3dHH.Reset();

		char *pLine = NULL;
		while (pLine = fout.GetLine()) {
			int nFeatNum = 0;
			int nRow, nCol;
			String strLabel = strtok(pLine, ": \t");
			pLine = strtok(NULL, ": \t");
			if (strLabel == "m_matVH")
			{
				// VH
				sscanf(pLine, "(num=%d*%d)", &nRow, &nCol);
				m_m3dVH.Reset(nRow, nCol, 2);
				m_m3dVH.Read(fout);
			}
			else if (strLabel == "m_matCH")
			{
				// CH
				sscanf(pLine, "(num=%d*%d)", &nRow, &nCol);
				m_m3dCH.Reset(nRow, nCol, 2);
				m_m3dCH.Read(fout);
			}
			else if (strLabel == "m_matHH")
			{
				sscanf(pLine, "(num=%d*%d)", &nRow, &nCol);
				m_m3dHH.Reset(nRow, nCol, 4);
				m_m3dHH.Read(fout);
			}
		}
	}
	void Model::WriteT(const char *pfilename)
	{
		File fout(pfilename, "wt");
		lout << "[Model] Write(txt) to " << pfilename << endl;

		fout.Print("m_vocabsize=%d\n", m_pVocab->GetSize());
		fout.Print("m_maxlen=%d\n", m_maxlen);
		fout.Print("m_hlayer=%d\n", m_hlayer);
		fout.Print("m_hnode=%d\n", m_hnode);
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

		// VH
		fout.Print("m_matVH: (num=%d*%d)\n", m_m3dVH.GetXDim(), m_m3dVH.GetYDim());
		m_m3dVH.Write(fout);

		// CH
		fout.Print("m_matCH: (num=%d*%d)\n", m_m3dCH.GetXDim(), m_m3dCH.GetYDim());
		m_m3dCH.Write(fout);

		// HH
		fout.Print("m_matHH: (num=%d*%d)\n", m_m3dHH.GetXDim(), m_m3dHH.GetYDim());
		m_m3dHH.Write(fout);
	}
	LogP Model::GetLogProb(VecShell<VocabID> &x, bool bNorm /* = true */)
	{
		LogP logProb = 0;
		for (int layer = 0; layer < m_hlayer; layer++) {
			AlgLayer alg(this, x, layer);
			alg.ForwardBackward(x.GetSize(), GetHiddenOrder(), GetEncodeLayerLimit());
			logProb += alg.GetLogSummation();
		}
		
		trf::Seq trfseq;
		trfseq.Set(x.GetBuf(), x.GetSize(), m_pVocab);
		/// Add all the ngram features
		logProb += FeatClusterSum(trfseq, 0, x.GetSize());

		if (bNorm)
			logProb = logProb - m_logz[x.GetSize()] + trf::Prob2LogP(m_pi[x.GetSize()]);
		return logProb;
	}
	LogP Model::ClusterSum(Seq &seq, int nPos, int nOrder)
	{
		return FeatClusterSum(seq.x, nPos, nOrder) + HiddenClusterSum(seq, nPos, nOrder);
	}
	LogP Model::FeatClusterSum(trf::Seq &x, int nPos, int nOrder)
	{
		return trf::Model::ClusterSum(x, nPos, nOrder);
	}
	LogP Model::HiddenClusterSum(Seq &seq, int nPos, int nOrder)
	{
		LogP LogSum = 0;

		// Word * hidden
		LogSum += SumVHWeight(m_m3dVH[seq.wseq()[nPos]], seq.h[nPos]);

		if (nPos == seq.GetLen() - nOrder) { // The last cluster
			for (int i = nPos + 1; i < seq.GetLen(); i++) {
				LogSum += SumVHWeight(m_m3dVH[seq.wseq()[i]], seq.h[i]);
			}
		}

		// Class * hidden
		if (m_m3dCH.GetSize() > 0) {
			LogSum += SumVHWeight(m_m3dCH[seq.cseq()[nPos]], seq.h[nPos]);

			if (nPos == seq.GetLen() - nOrder) { // The last cluster
				for (int i = nPos + 1; i < seq.GetLen(); i++) {
					LogSum += SumVHWeight(m_m3dCH[seq.cseq()[i]], seq.h[i]);
				}
			}
		}

		// Hidden * Hidden
		if (nOrder > 1) { // if order=1, then no HH matrix
			LogSum += SumHHWeight(m_m3dHH, seq.h[nPos], seq.h[nPos + 1]);

			if (nPos == seq.GetLen() - nOrder) { // The last cluster
				for (int i = nPos + 1; i < seq.GetLen() - 1; i++)
					LogSum += SumHHWeight(m_m3dHH, seq.h[i], seq.h[i + 1]);
			}
		}


		return LogSum;
	}
	LogP Model::LayerClusterSum(Seq &seq, int nlayer, int nPos, int nOrder)
	{
		LogP LogSum = 0;

		// Word * hidden
		LogSum += SumVHWeight(m_m3dVH[seq.wseq()[nPos]], seq.h[nPos], nlayer);

		if (nPos == seq.GetLen() - nOrder) { // The last cluster
			for (int i = nPos + 1; i < seq.GetLen(); i++) {
				LogSum += SumVHWeight(m_m3dVH[seq.wseq()[i]], seq.h[i], nlayer);
			}
		}

		// Class * hidden
		if (m_m3dCH.GetSize() > 0) {
			LogSum += SumVHWeight(m_m3dCH[seq.cseq()[nPos]], seq.h[nPos], nlayer);

			if (nPos == seq.GetLen() - nOrder) { // The last cluster
				for (int i = nPos + 1; i < seq.GetLen(); i++) {
					LogSum += SumVHWeight(m_m3dCH[seq.cseq()[i]], seq.h[i], nlayer);
				}
			}
		}

		// Hidden * Hidden
		if (nOrder > 1) { // if order=1, then no HH matrix
			LogSum += SumHHWeight(m_m3dHH, seq.h[nPos], seq.h[nPos + 1], nlayer);

			if (nPos == seq.GetLen() - nOrder) { // The last cluster
				for (int i = nPos + 1; i < seq.GetLen() - 1; i++)
					LogSum += SumHHWeight(m_m3dHH, seq.h[i], seq.h[i + 1], nlayer);
			}
		}


		return LogSum;
	}
	double Model::ExactNormalize(int nLen)
	{
		int nMaxOrder = max(GetMaxOrder(), GetHiddenOrder()); ///< max-order
		int nIterDim = min(nMaxOrder, nLen);


		/* as for exact Z_1 is need in joint SA algorithm.
		Calculate Z_1 using a different way
		*/
		if (nLen == 1) {
			double dLogSum = trf::LogP_zero;
			for (VocabID x = m_pVocab->IterBeg(); x <= m_pVocab->IterEnd(); x++) {
				trf::Seq xseq;
				VocabID cid = m_pVocab->GetClass(x);
				xseq.Set(&x, 1, m_pVocab);
				double d1 = FeatClusterSum(xseq, 0, 1);
				double d2 = 0;
				for (int k = 0; k < m_hlayer * m_hnode; k++) {
					/* After introducing CHmat, revise the equation !!! */
					if (cid != trf::VocabID_none && m_m3dCH.GetSize() > 0) {
						d2 += trf::Log_Sum(m_m3dVH[x][k][0] + m_m3dCH[cid][k][0], m_m3dVH[x][k][1] + m_m3dCH[cid][k][1]);
					}
					else { // if cid == VocabID_none, it means on class infromation
						d2 += trf::Log_Sum(m_m3dVH[x][k][0], m_m3dVH[x][k][1]);
					}
				}
				dLogSum = trf::Log_Sum(dLogSum, d1 + d2);
			}
			m_logz[nLen] = dLogSum;
		}
		else {
			int nEncoderLimit = GetEncodeNodeLimit();
			// forward-backward
			m_nodeCal.ForwardBackward(nLen, nMaxOrder, nEncoderLimit);

			m_logz[nLen] = m_nodeCal.GetLogSummation();
		}

		return m_logz[nLen];

	}
	void Model::ExactNormalize()
	{
		for (int len = 1; len <= m_maxlen; len++) {
			ExactNormalize(len);
			m_zeta[len] = m_logz[len] - m_logz[1];
			//lout << " logZ[" << len << "] = " << m_logz[len] << endl;
		}
	}
	LogP Model::GetMarginalLogProb(int nLen, int nPos, Seq &sub, bool bNorm /* = true */)
	{
		// Forward-backward need be calculate

		if (nPos + sub.GetLen() > nLen) {
			lout_error("[Model] GetMarginalLogProb: nPos(" << nPos << ")+nOrder(" << sub.GetLen() << ") > seq.len(" << nLen << ")!!");
		}

		// encode the sub sequence
		Vec<int> nsub(sub.GetLen());
		EncodeNode(nsub, sub);

		LogP dSum = m_nodeCal.GetMarginalLogProb(nPos, nsub.GetBuf(), nsub.GetSize());

		return (bNorm) ? dSum - m_logz[nLen] : dSum;
	}

	void Model::GetNodeExp(double *pExp, Prob *pLenProb/* = NULL*/)
	{
		if (pLenProb == NULL)
			pLenProb = m_pi.GetBuf();
		VecShell<double> exp(pExp, GetParamNum());
		Vec<double> expTemp(GetParamNum());

		double *p = expTemp.GetBuf();
		VecShell<double> featexp;
		Mat3dShell<double> VHexp, CHexp, HHexp;
		BufMap(p, featexp, VHexp, CHexp, HHexp);

		exp.Fill(0);
		for (int len = 1; len <= m_maxlen; len++) {

			int nMaxOrder = max(GetMaxOrder(), GetHiddenOrder()); ///< max-order
			m_nodeCal.ForwardBackward(len, nMaxOrder, GetEncodeNodeLimit());

			GetNodeExp(len, featexp, VHexp, CHexp, HHexp);
			// 			GetNodeExp_feat(len, featexp);
			// 			GetNodeExp_VH(len, VHexp);
			// 			GetNodeExp_HH(len, HHexp);

			for (int i = 0; i < exp.GetSize(); i++) {
				exp[i] += pLenProb[len] * expTemp[i];
			}
		}
	}
	void Model::GetNodeExp(int nLen, double *pExp)
	{
		VecShell<double> featexp;
		Mat3dShell<double> VHexp, CHexp, HHexp;
		BufMap(pExp, featexp, VHexp, CHexp, HHexp);
		GetNodeExp(nLen, featexp, VHexp, CHexp, HHexp);
	}
	void Model::GetNodeExp(int nLen, VecShell<double> featexp,
		Mat3dShell<double> VHexp, Mat3dShell<double> CHexp, Mat3dShell<double> HHexp)
	{
		// make sure the forward-backward is performed.
		featexp.Fill(0);
		VHexp.Fill(0);
		CHexp.Fill(0);
		HHexp.Fill(0);

		//int nMaxOrder = m_nodeCal.m_nOrder;
		int nClusterNum = nLen - m_nodeCal.m_nOrder + 1;
		int nClusterDim = m_nodeCal.m_nOrder;
		if (nClusterNum < 1) {
			nClusterNum = 1;
			nClusterDim = nLen;
		}

		Vec<int> nseq(nLen);
		Seq seq(nLen, m_hlayer, m_hnode);

		// circle for the position pos
		for (int pos = 0; pos < nClusterNum; pos++) {
			// ergodic the cluster
			trf::VecIter iter(nseq.GetBuf() + pos, nClusterDim, 0, GetEncodeNodeLimit() - 1);
			while (iter.Next()) {
				DecodeNode(nseq, seq, pos, nClusterDim); /// decoder to x and h
				Prob prob = trf::LogP2Prob(m_nodeCal.GetMarginalLogProb(pos, nseq.GetBuf()+pos, nClusterDim, m_logz[nLen]));

				//////////////////////////////////////////////////////////////////////////
				// the cluster before the last one
				//////////////////////////////////////////////////////////////////////////
				Array<int> afeat;
				for (int n = 1; n <= nClusterDim; n++) {
					m_pFeat->Find(afeat, seq.x, pos, n);
				}
				for (int i = 0; i < afeat.GetNum(); i++) {
					featexp[afeat[i]] += prob;
				}

				VocabID x = seq.wseq()[pos];
				for (int k = 0; k < m_hlayer*m_hnode; k++) {
					VHexp[x][k][(int)(seq.h[pos][k])] += prob;
				}
				if (m_pVocab->GetClassNum() > 0) {
					VocabID c = seq.cseq()[pos];
					for (int k = 0; k < m_hlayer*m_hnode; k++) {
						CHexp[c][k][(int)(seq.h[pos][k])] += prob;
					}
				}
				if (nClusterDim > 1) {
					for (int l = 0; l < m_hlayer; l++) {
						for (int a = 0; a < m_hnode; a++) {
							for (int b = 0; b < m_hnode; b++) {
								HHexp[l*m_hnode + a][b][HHMap(seq.h[pos][l*m_hnode + a], seq.h[pos + 1][l*m_hnode + b])] += prob;
							}
						}
					}
					
				}


				//////////////////////////////////////////////////////////////////////////
				// the last cluster
				//////////////////////////////////////////////////////////////////////////
				if (pos == nClusterNum - 1) {
					afeat.Clean();
					for (int ii = 1; ii < nClusterDim; ii++) { // position ii
						for (int n = 1; n <= nClusterDim - ii; n++) { // order n
							m_pFeat->Find(afeat, seq.x, pos + ii, n);
						}
					}
					for (int i = 0; i < afeat.GetNum(); i++) {
						featexp[afeat[i]] += prob;
					}

					for (int ii = 1; ii < nClusterDim; ii++) {
						VocabID x = seq.wseq()[pos+ii];
						for (int k = 0; k < m_hlayer*m_hnode; k++) {
							VHexp[x][k][seq.h[pos+ii][k]] += prob;
						}
					}
					if (m_pVocab->GetClassNum() > 0) {
						for (int ii = 1; ii < nClusterDim; ii++) {
							VocabID c = seq.cseq()[pos+ii];
							for (int k = 0; k < m_hlayer*m_hnode; k++) {
								CHexp[c][k][seq.h[pos+ii][k]] += prob;
							}
						}
					}
					for (int ii = 1; ii < nClusterDim - 1; ii++) {
						for (int l = 0; l < m_hlayer; l++) {
							for (int a = 0; a < m_hnode; a++) {
								for (int b = 0; b < m_hnode; b++) {
									HHexp[l*m_hnode + a][b][HHMap(seq.h[pos + ii][l*m_hnode + a], seq.h[pos + ii + 1][l*m_hnode + b])] += prob;
								}
							}
						}
					}
				}
			}
		}
	}

	void Model::GetHiddenExp(VecShell<VocabID> x, double *pExp)
	{

		VecShell<double> featexp;
		Mat3dShell<double> VHexp, CHexp, HHexp;
		BufMap(pExp, featexp, VHexp, CHexp, HHexp);


		int nLen = x.GetSize(); ///< length
		int nMaxOrder = GetHiddenOrder(); ///< max-order

		for (int layer = 0; layer < m_hlayer; layer++) {
			AlgLayer fb(this, x, layer);
			// forward-backward
			fb.ForwardBackward(nLen, nMaxOrder, GetEncodeLayerLimit());
			// get the normalization constant
			LogP logz = fb.GetLogSummation();
			// get the exp
			GetLayerExp(fb, layer, VHexp, CHexp, HHexp, logz);
		}

		//get the feature expectation
		trf::Seq trfseq(nLen);
		trfseq.Set(x.GetBuf(), nLen, m_pVocab);
		trf::Model::FeatCount(trfseq, featexp.GetBuf());
	}

	void Model::GetLayerExp(AlgLayer &fb, int nLayer,
		Mat3dShell<double> &VHexp, Mat3dShell<double> &CHexp, Mat3dShell<double> &HHexp, LogP logz /* = 0 */)
	{
		/* Don't clean the buffer!!!! */
		//int nMaxOrder = GetHiddenOrder();
		int nLen = fb.m_nLen;
		int nClusterNum = nLen - fb.m_nOrder + 1;
		int nClusterDim = fb.m_nOrder;
		if (nClusterNum < 1) {
			nClusterNum = 1;
			nClusterDim = nLen;
		}

		Vec<int> hseq(nLen);
		Mat<HValue> h(nLen, m_hlayer * m_hnode);
		for (int pos = 0; pos < nClusterNum; pos++) {
			// ergodic the cluster
			trf::VecIter iter(hseq.GetBuf() + pos, nClusterDim, 0, GetEncodeLayerLimit() - 1);
			while (iter.Next()) {
				DecodeLayer(hseq, h, nLayer, pos, nClusterDim);
				Prob prob = trf::LogP2Prob(fb.GetMarginalLogProb(pos, hseq.GetBuf()+pos, nClusterDim, logz)); // the prob of current cluster

				// the cluster before the last one
				VocabID x = fb.m_seq.wseq()[pos];
				for (int k = nLayer*m_hnode; k < nLayer*m_hnode+m_hnode; k++) {
					VHexp[x][k][h[pos][k]] += prob;
				}
				if (m_pVocab->GetClassNum() > 0) {
					VocabID c = fb.m_seq.cseq()[pos];
					for (int k = nLayer*m_hnode; k < nLayer*m_hnode + m_hnode; k++) {
						CHexp[c][k][h[pos][k]] += prob;
					}
				}
				if (nClusterDim > 1) {
					for (int a = 0; a < m_hnode; a++) {
						for (int b = 0; b < m_hnode; b++) {
							HHexp[nLayer*m_hnode + a][b][HHMap(h[pos][nLayer*m_hnode + a], h[pos + 1][nLayer*m_hnode + b])] += prob;
						}
					}
				}

				// the last cluster
				if (pos == nClusterNum - 1) {
					for (int ii = 1; ii < nClusterDim; ii++) {
						VocabID x = fb.m_seq.wseq()[pos + ii];
						for (int k = nLayer*m_hnode; k < nLayer*m_hnode + m_hnode; k++) {
							VHexp[x][k][h[pos + ii][k]] += prob;
						}
						if (m_pVocab->GetClassNum() > 0) {
							VocabID c = fb.m_seq.cseq()[pos + ii];
							for (int k = nLayer*m_hnode; k < nLayer*m_hnode + m_hnode; k++) {
								CHexp[c][k][h[pos + ii][k]] += prob;
							}
						}
					}
					for (int ii = 1; ii < nClusterDim - 1; ii++) {
						for (int a = 0; a < m_hnode; a++) {
							for (int b = 0; b < m_hnode; b++) {
								HHexp[nLayer*m_hnode + a][b][HHMap(h[pos+ii][nLayer*m_hnode + a], h[pos + ii + 1][nLayer*m_hnode + b])] += prob;
							}
						}
					}
				}
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
			seq.Reset(nNewLen, seq.GetHlayer(), seq.GetHnode());
			LogP Q = ProposeH0(seq.h[nNewLen - 1], seq, nNewLen - 1, true);
			LogP R = ProposeC0(seq.cseq()[nNewLen - 1], seq, nNewLen - 1, true);
			LogP G = SampleW(seq, nNewLen - 1);
			LogP logpnew = GetLogProb(seq);

			logpAcc = (j2 - j1) + logpnew - (logpold + Q + R + G);
		}
		else if (nNewLen == nOldLen - 1) {
			LogP logpold = GetLogProb(seq);
			LogP Q = ProposeH0(seq.h[nOldLen - 1], seq, nOldLen - 1, false);
			LogP R = ProposeC0(seq.cseq()[nOldLen - 1], seq, nOldLen - 1, false);
			LogP G = SampleW(seq, nOldLen - 1, false);

			seq.Reset(nNewLen, seq.GetHlayer(), seq.GetHnode());
			LogP logpnew = GetLogProb(seq);

			logpAcc = (j2 - j1) + logpnew + Q + R + G - logpold;
		}
		else if (nNewLen != nOldLen){
			lout_error("[Model] Sample: nNewLen(" << nNewLen << ") and nOldLen(" << nOldLen << ")");
		}


		if (trf::Acceptable(trf::LogP2Prob(logpAcc))) {
			seq.Reset(nNewLen, seq.GetHlayer(), seq.GetHnode());
			m_nLenJumpAccTimes++;
		}
		else {
			seq.Reset(nOldLen, seq.GetHlayer(), seq.GetHnode());
		}
		m_nLenJumpTotalTime++;

	}
	void Model::MarkovMove(Seq &seq)
	{
		/* Gibbs sampling */
		SampleHAndCGivenX(seq);
		for (int nPos = 0; nPos < seq.GetLen(); nPos++) {
			SampleC(seq, nPos);
			SampleW(seq, nPos);
		}
		//SampleHAndCGivenX(seq);
	}

	LogP Model::ProposeLength(int nOld, int &nNew, bool bSample)
	{
		if (bSample) {
			nNew = trf::LineSampling(m_matLenJump[nOld].GetBuf(), m_maxlen + 1);
		}

		return trf::Prob2LogP(m_matLenJump[nOld][nNew]);
	}
	LogP Model::ProposeH0(VecShell<HValue> &hi, Seq &seq, int nPos, bool bSample)
	{
		/* Note:
		The nPos may be larger than the length of seq. i.e nPos >= seq.GetLen();
		As we may want to propose a new position over the sequence.
		*/

		if (nPos + 1 > seq.GetLen()) {
			seq.Reset(nPos + 1, seq.GetHlayer(), seq.GetHnode());
		}

		Vec<LogP> logps(m_hlayer*m_hnode);
		ProposeHProbs(logps, seq, nPos);


		/* Sample */
		if (bSample) {
			for (int i = 0; i < logps.GetSize(); i++) {
				hi[i] = trf::Acceptable(trf::LogP2Prob(logps[i])) ? 1.0f : 0.0f;
			}
		}

		/* Get The probs */
		LogP resLogp = GetConditionalProbForH(hi, logps);


		return resLogp;
	}
	LogP Model::ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample)
	{
		/* if there are no class, then return 0 */
		if (m_pVocab->GetClassNum() == 0) {
			ci = trf::VocabID_none;
			return 0;
		}

		Vec<LogP> vlogps(m_pVocab->GetClassNum());
		ProposeCProbs(vlogps, seq, nPos);

		if (bSample) {
			ci = trf::LogLineSampling(vlogps.GetBuf(), vlogps.GetSize());
		}

		return vlogps[ci];
	}
	void Model::ProposeHProbs(VecShell<LogP> &logps, Seq &seq, int nPos, bool bConsiderXandC /*=false*/)
	{
		logps.Fill(0);
		Mat<LogP> matLogp(m_hlayer*m_hnode, 2); /// save the logp of 0 or 1 for each hidden vairables
		matLogp.Fill(0);

		// HH connection
		if (nPos - 1 >= 0 && nPos - 1 <= seq.GetLen() - 1) {
			for (int l = 0; l < m_hlayer; l++) {
				for (int i = 0; i < m_hnode; i++) {
					HValue curh = seq.h[nPos - 1][l*m_hnode + i];
					for (int j = 0; j < m_hnode; j++) {
						matLogp.Get(l*m_hnode + j, 0) += m_m3dHH.Get(l*m_hnode + i, j, HHMap(curh, 0));
						matLogp.Get(l*m_hnode + j, 1) += m_m3dHH.Get(l*m_hnode + i, j, HHMap(curh, 1));
					}
				}
			}
		}
		if (nPos + 1 <= seq.GetLen() - 1) {
			for (int l = 0; l < m_hlayer; l++) {
				for (int i = 0; i < m_hnode; i++) {
					HValue curh = seq.h[nPos + 1][l*m_hnode + i];
					for (int j = 0; j < m_hnode; j++) {
						matLogp.Get(l*m_hnode + j, 0) += m_m3dHH.Get(l*m_hnode + j, i, HHMap(0, curh));
						matLogp.Get(l*m_hnode + j, 1) += m_m3dHH.Get(l*m_hnode + j, i, HHMap(1, curh));
					}
				}
			}
		}

		if (bConsiderXandC) {
			/* Consider the VH matrix */
			for (int i = 0; i < m_hlayer*m_hnode; i++) {
				matLogp[i][0] += m_m3dVH[seq.wseq()[nPos]][i][0];
				matLogp[i][1] += m_m3dVH[seq.wseq()[nPos]][i][1];
			}
			if (m_m3dCH.GetSize() > 0) {
				/* Consider the CH matrix */
				for (int i = 0; i < m_hlayer*m_hnode; i++) {
					matLogp[i][0] += m_m3dCH[seq.cseq()[nPos]][i][0];
					matLogp[i][1] += m_m3dCH[seq.cseq()[nPos]][i][1];
				}
			}
		}

		/*
		Get Probs
		*/
		for (int i = 0; i < m_hlayer*m_hnode; i++) {
			//logps[i] = logps[i] - Log_Sum(logps[i], 0);
			logps[i] = matLogp[i][1] - trf::Log_Sum(matLogp[i][1], matLogp[i][0]);
		}
	}
	void Model::ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos)
	{
		VocabID savecid = seq.cseq()[nPos];
		for (int cid = 0; cid < m_pVocab->GetClassNum(); cid++) {
			seq.cseq()[nPos] = cid;
			logps[cid] = GetReducedModelForC(seq, nPos);
		}
		seq.cseq()[nPos] = savecid;
		trf::LogLineNormalize(logps.GetBuf(), m_pVocab->GetClassNum());
	}
	LogP Model::GetReducedModelForH(Seq &seq, int nPos)
	{
		// Only consider the HH-matrix, as VH matrix has been considered in GetLogWeightSumForW
		LogP logSum = 0;
		// Hidden * Hidden
		for (int i = max(0, nPos - 1); i <= min(seq.GetLen() - 2, nPos); i++) {
			logSum += SumHHWeight(m_m3dHH, seq.h[i], seq.h[i + 1]);
		}
		return logSum;
	}
	LogP Model::GetReducedModelForC(Seq &seq, int nPos)
	{
		// class features
		LogP logSum = trf::Model::GetReducedModelForC(seq.x, nPos);

		// CH
		if (m_m3dCH.GetSize() > 0) {
			logSum += SumVHWeight(m_m3dCH[seq.cseq()[nPos]], seq.h[nPos]);
		}

		return logSum;
	}
	LogP Model::GetReducedModelForW(Seq &seq, int nPos)
	{
		// word features
		LogP logSum = trf::Model::GetReducedModelForW(seq.x, nPos);
		// VH
		logSum += SumVHWeight(m_m3dVH[seq.wseq()[nPos]], seq.h[nPos]);
		return logSum;
	}
	LogP Model::GetConditionalProbForH(VecShell<HValue> &hi, VecShell<LogP> &logps)
	{
		/* Get The probs */
		LogP resLogp = 0;
		for (int i = 0; i < hi.GetSize(); i++) {
			resLogp += (hi[i] == 0) ? trf::Log_Sub(0, logps[i]) : logps[i];
		}

		return resLogp;
	}
	LogP Model::GetMarginalProbOfC(Seq &seq, int nPos)
	{
		LogP resLogp = trf::LogP_zero;

		Array<VocabID> *pXs = m_pVocab->GetWord(seq.cseq()[nPos]);

		VocabID saveX = seq.wseq()[nPos];
		for (int i = 0; i < pXs->GetNum(); i++) {
			seq.wseq()[nPos] = pXs->Get(i);
			/* Only need to calculate the summation of weight depending on x[nPos], c[nPos] */
			/* used to sample the c_i, fixed h */
			resLogp = trf::Log_Sum(resLogp, GetReducedModelForW(seq, nPos) + GetReducedModelForC(seq, nPos));
			//resLogp = Log_Sum(resLogp, GetLogProb(seq, false));
		}
		seq.wseq()[nPos] = saveX;

		return resLogp;
	}
	void Model::SampleC(Seq &seq, int nPos)
	{
		if (m_pVocab->GetClassNum() == 0) {
			seq.cseq()[nPos] = trf::VocabID_none;
			return;
		}

		/* Sample C0 */
		Vec<LogP> vlogps_c(m_pVocab->GetClassNum());
		ProposeCProbs(vlogps_c, seq, nPos);
		VocabID ci = seq.cseq()[nPos];
		VocabID C0 = trf::LogLineSampling(vlogps_c.GetBuf(), vlogps_c.GetSize());
		LogP logpRi = vlogps_c[ci];
		LogP logpR0 = vlogps_c[C0];


		/* Calculate the probability p_t(h, c) */
		seq.cseq()[nPos] = ci;
		LogP Logp_ci = GetMarginalProbOfC(seq, nPos);
		seq.cseq()[nPos] = C0;
		LogP Logp_C0 = GetMarginalProbOfC(seq, nPos);

		LogP acclogp = logpRi + Logp_C0 - (logpR0 + Logp_ci);

		m_nSampleHTotalTimes++;
		if (trf::Acceptable(trf::LogP2Prob(acclogp))) {
			m_nSampleHAccTimes++;
			seq.cseq()[nPos] = C0;
		}
		else {
			seq.cseq()[nPos] = ci;
		}
	}
	LogP Model::SampleW(Seq &seq, int nPos, bool bSample/* = true*/)
	{
		/*
		The function calculate G(w_i| w_{other}, c, h)
		if bSample is true, draw a sample for w_i;
		otherwise, only calcualte the conditional probability.
		*/
		if (nPos >= seq.GetLen()) {
			lout_error("[Model] SampleH: the nPos(" << nPos << ") > the length of sequence(" << seq.GetLen() << ")");
		}

		Array<VocabID> *pXs = m_pVocab->GetWord(seq.cseq()[nPos]);
		Array<LogP> aLogps;

		VocabID nSaveX = seq.wseq()[nPos]; // save x[nPos]
		for (int i = 0; i < pXs->GetNum(); i++) {
			seq.wseq()[nPos] = pXs->Get(i);
			/* To reduce the computational cost, instead of GetLogProb,
			we just need to calculate the summation of weight depending on w[nPos]
			*/
			aLogps[i] = GetReducedModelForW(seq, nPos);
		}
		trf::LogLineNormalize(aLogps, pXs->GetNum());

		int idx;
		if (bSample) {
			/* sample a value for x[nPos] */
			idx = trf::LogLineSampling(aLogps, pXs->GetNum());
			seq.wseq()[nPos] = pXs->Get(idx);
		}
		else {
			idx = pXs->Find(nSaveX); // find nSave in the array.
			seq.wseq()[nPos] = nSaveX;
			if (idx == -1) {
				lout_error("Can't find the VocabID(" << nSaveX << ") in the array.\n"
					<< "This may beacuse word(" << nSaveX << ") doesnot belongs to class(" << seq.cseq()[nPos] << ")");
			}
		}

		return aLogps[idx];
	}
	LogP Model::SampleHAndCGivenX(Seq &seq, MatShell<HValue> *tagH /* = NULL */)
	{
		LogP totallogProb = 0;

		/* set class */
		m_pVocab->GetClass(seq.cseq(), seq.wseq(), seq.GetLen());

		/* sample h */
		for (int nPos = 0; nPos < seq.GetLen(); nPos++) {
			Vec<LogP> vlogps_h(m_hlayer * m_hnode);
			ProposeHProbs(vlogps_h, seq, nPos, true);

			Vec<HValue> hsample(m_hlayer * m_hnode);
			if (tagH) {
				hsample.Copy((*tagH)[nPos]);
			}
			else { /* sampling */
				for (int i = 0; i < hsample.GetSize(); i++) {
					hsample[i] = trf::Acceptable(trf::LogP2Prob(vlogps_h[i])) ? 1.0f : 0.0f;
				}

			}
			seq.h[nPos] = hsample;

			LogP logprob = GetConditionalProbForH(hsample, vlogps_h);
			totallogProb += logprob;
		}
		return totallogProb;
	}

	void Model::RandSeq(Seq &seq, int nLen /* = -1 */)
	{
		if (nLen == -1) {
			seq.Reset(rand() % GetMaxLen() + 1, m_hlayer, m_hnode);
		}
		else {
			seq.Reset(nLen, m_hlayer, m_hnode);
		}

		/* randomly set h*/
		for (int i = 0; i < seq.h.GetRow(); i++) {
			for (int k = 0; k < seq.h.GetCol(); k++) {
				seq.h[i][k] = rand() % 2;
			}
		}

		seq.x.Random(m_pVocab);
	}
	void Model::RandHidden(Seq &seq)
	{
		/* randomly set h*/
		for (int i = 0; i < seq.h.GetRow(); i++) {
			for (int k = 0; k < seq.h.GetCol(); k++) {
				seq.h[i][k] = rand() % 2;
			}
		}
	}

	int Model::EncodeNode(VocabID xi, VocabID ci, VecShell<HValue> &hi)
	{
		int hnum = EncodeHidden(hi);

		return hnum * m_pVocab->GetSize() + xi;
	}
	void Model::EncodeNode(VecShell<int> &vn, Seq &seq, int nPos /* = 0 */, int nDim /* = -1 */)
	{
		nDim = (nDim == -1) ? seq.GetLen() - nPos : nDim;
		for (int i = nPos; i < nPos + nDim; i++) {
			vn[i] = EncodeNode(seq.wseq()[i], seq.cseq()[i], seq.h[i]);
		}
	}
	void Model::DecodeNode(int n, VocabID &xi, VocabID &ci, VecShell<HValue> &hi)
	{
		int hnum = n / m_pVocab->GetSize();

		xi = n % m_pVocab->GetSize();
		ci = m_pVocab->GetClass(xi);
		DecodeHidden(hnum, hi);
	}
	void Model::DecodeNode(VecShell<int> &vn, Seq &seq, int nPos /* = 0 */, int nDim /* = -1 */)
	{
		nDim = (nDim == -1) ? vn.GetSize() - nPos : nDim;
		for (int i = nPos; i < nPos + nDim; i++) {
			DecodeNode(vn[i], seq.wseq()[i], seq.cseq()[i], seq.h[i]);
		}
	}
	int Model::GetEncodeNodeLimit() const
	{
		return GetEncodeHiddenLimit() * m_pVocab->GetSize();
	}
	int Model::EncodeHidden(VecShell<HValue> hi)
	{
		int hnum = 0;
		for (int i = 0; i < hi.GetSize(); i++) {
			hnum += (int)hi[i] * (1 << i);
		}

		return hnum;
	}
	void Model::DecodeHidden(int n, VecShell<HValue> hi)
	{
		for (int i = 0; i < hi.GetSize(); i++) {
			hi[i] = n % 2;
			n >>= 1;
		}
	}
	void Model::DecodeHidden(VecShell<int> &vn, Mat<HValue> &h, int nPos /* = 0 */, int nDim /* = -1 */)
	{
		nDim = (nDim == -1) ? vn.GetSize() - nPos : nDim;
		for (int i = nPos; i < nPos + nDim; i++) {
			DecodeHidden(vn[i], h[i]);
		}
	}
	int Model::GetEncodeHiddenLimit() const
	{
		/* if the m_hnode >= 32, the value over the maxiume number of int*/
		if (m_hnode >= 30) {
			lout_error("[Model] GetEncodeHiddenLimit: overflow! m_hnode = " << m_hnode);
		}
		return 1 << m_hlayer * m_hnode;
	}
	void Model::DecodeLayer(VecShell<int> &vn, Mat<HValue> &h, int layer, int nPos /* = 0 */, int nDim /* = -1 */)
	{
		nDim = (nDim == -1) ? vn.GetSize() - nPos : nDim;
		for (int i = nPos; i < nPos + nDim; i++) {
			DecodeHidden(vn[i], h[i].GetSub(layer*m_hnode, m_hnode));
		}
	}
	int Model::GetEncodeLayerLimit() const
	{
		return 1 << m_hnode;
	}

	void Model::FeatCount(Seq &seq, VecShell<double> featcount, Mat3dShell<double> VHcount, Mat3dShell<double> CHcount, Mat3dShell<double> HHcount, double dadd /* = 1 */)
	{
		trf::Model::FeatCount(seq.x, featcount.GetBuf(), dadd);

		HiddenFeatCount(seq, VHcount, CHcount, HHcount, dadd);
	}
	void Model::HiddenFeatCount(Seq &seq, Mat3dShell<double> VHcount, Mat3dShell<double> CHcount, Mat3dShell<double> HHcount, double dadd /* = 1 */)
	{
		/* VH count */
		for (int i = 0; i < seq.GetLen(); i++) {
			for (int k = 0; k < m_hlayer*m_hnode; k++) {
				VHcount[seq.wseq()[i]][k][seq.h[i][k]] += dadd;
			}
		}

		/* CH count */
		if (m_pVocab->GetClassNum() > 0) {
			for (int i = 0; i < seq.GetLen(); i++) {
				for (int k = 0; k < m_hlayer*m_hnode; k++) {
					CHcount[seq.cseq()[i]][k][seq.h[i][k]] += dadd;
				}
			}
		}

		/* HH count */
		for (int i = 0; i < seq.GetLen() - 1; i++) {
			for (int l = 0; l < m_hlayer; l++) {
				for (int a = 0; a < m_hnode; a++) {
					for (int b = 0; b < m_hnode; b++) {
						HHcount.Get(l * m_hnode + a, b, HHMap(seq.h.Get(i, l*m_hnode + a), seq.h.Get(i + 1, l*m_hnode + b))) += dadd;
					}
				}
			}
			
		}
	}
	void Model::FeatCount(Seq &seq, VecShell<double> count, double dadd /* = 1 */)
	{
		VecShell<double> featcount;
		Mat3dShell<double> VHcount, CHcount, HHcount;
		BufMap(count.GetBuf(), featcount, VHcount, CHcount, HHcount);
		FeatCount(seq, featcount, VHcount, CHcount, HHcount, dadd);
	}

	PValue Model::SumVHWeight(MatShell<PValue> m, VecShell<HValue> h)
	{
		PValue dsum = 0;
		for (int i = 0; i < h.GetSize(); i++) {
			dsum += m[i][(int)h[i]];
		}
		return dsum;
	}
	PValue Model::SumHHWeight(Mat3dShell<PValue> m, VecShell<HValue> h1, VecShell<HValue> h2)
	{
		PValue dsum = 0;

		for (int k = 0; k < m_hlayer; k++) {
			for (int i = 0; i < m_hnode; i++) {
				for (int j = 0; j < m_hnode; j++)
				{
					dsum += m.Get(k*m_hnode + i, j, HHMap(h1[k*m_hnode + i], h2[k*m_hnode + j]));
				}
			}
		}
		return dsum;
	}
	PValue Model::SumVHWeight(MatShell<PValue> m, VecShell<HValue> h, int layer)
	{
		PValue dsum = 0;
		for (int i = layer * m_hnode; i < layer *m_hnode + m_hnode; i++) {
			dsum += m[i][(int)h[i]];
		}
		return dsum;
	}
	PValue Model::SumHHWeight(Mat3dShell<PValue> m, VecShell<HValue> h1, VecShell<HValue> h2, int layer)
	{
		PValue dsum = 0;

		int k = layer;
		for (int i = 0; i < m_hnode; i++) {
			for (int j = 0; j < m_hnode; j++)
			{
				dsum += m.Get(k*m_hnode + i, j, HHMap(h1[k*m_hnode + i], h2[k*m_hnode + j]));
			}
		}
		
		return dsum;
	}



	/************************************************************************/
	/* Forward-backward class                                               */
	/************************************************************************/
	AlgNode::AlgNode(Model *p)
	{
		m_pModel = p;
	}
	LogP AlgNode::ClusterSum(int *pSeq, int nLen, int nPos, int nOrder)
	{
		m_seq.Reset(nLen, m_pModel->m_hlayer, m_pModel->m_hnode);
		m_pModel->DecodeNode(VecShell<int>(pSeq, nLen), m_seq, nPos, nOrder);
		return m_pModel->ClusterSum(m_seq, nPos, nOrder);
	}

	AlgLayer::AlgLayer(Model *p, VecShell<VocabID> x, int nlayer)
	{
		m_pModel = p;
		m_nlayer = nlayer;
		m_seq.Reset(x.GetSize(), p->m_hlayer, p->m_hnode);
		m_seq.x.Set(x.GetBuf(), x.GetSize(), p->GetVocab());
	}
	LogP AlgLayer::ClusterSum(int *pSeq, int nLen, int nPos, int nOrder)
	{
		m_pModel->DecodeLayer(VecShell<int>(pSeq, nLen), m_seq.h, m_nlayer, nPos, nOrder);
		return m_pModel->LayerClusterSum(m_seq, m_nlayer, nPos, nOrder);
	}
}