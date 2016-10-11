#include "hrf-code-exam.h"

namespace hrf
{
	void ModelExam::SetValueAll(PValue v)
	{
		Vec<PValue> vParams(pm->GetParamNum());
		vParams.Fill(v);
		pm->SetParam(vParams.GetBuf());
	}
	void ModelExam::SetValueRand()
	{
		Vec<PValue> vParams(pm->GetParamNum());
		for (int i = 0; i < vParams.GetSize(); i++)
			vParams[i] = 1.0 * rand() / RAND_MAX - 0.5;
		pm->SetParam(vParams.GetBuf());
	}
	void ModelExam::TestNormalization(int nLen)
	{
		lout << "[ModelExam] Test Normalization (nlen=" << nLen << ")" << endl;
		Vec<int> nodeSeq(nLen);
		Seq seq(nLen, pm->m_hlayer, pm->m_hnode);

		LogP logz1 = pm->ExactNormalize(nLen);
		LogP logz2 = trf::LogP_zero;
		trf::VecIter nodeIter(nodeSeq.GetBuf(), nLen, 0, pm->GetEncodeNodeLimit() - 1);
		while (nodeIter.Next()) {
			pm->DecodeNode(nodeSeq, seq);
			logz2 = trf::Log_Sum(logz2, pm->GetLogProb(seq, false));
		}
		lout_variable(logz1);
		lout_variable(logz2);
	}
	void ModelExam::TestExpectation(int nLen)
	{
		lout << "[ModelExam] Test Expectation (nlen=" << nLen << ")" << endl;
		Vec<int> nodeSeq(nLen);
		Seq seq(nLen, pm->m_hlayer, pm->m_hnode);

		pm->ExactNormalize(nLen);

		Vec<double> exp1(pm->GetParamNum());
		Vec<double> exp2(pm->GetParamNum());
		exp1.Fill(0);
		exp2.Fill(0);

		pm->GetNodeExp(nLen, exp1.GetBuf());
		trf::VecIter nodeIter(nodeSeq.GetBuf(), nLen, 0, pm->GetEncodeNodeLimit() - 1);
		while (nodeIter.Next()) {
			pm->DecodeNode(nodeSeq, seq);
			LogP lp = pm->GetLogProb(seq, true);
			pm->FeatCount(seq, exp2, trf::LogP2Prob(lp));
		}

		for (int i = 0; i < pm->GetParamNum(); i++) {
			if (fabs(exp1[i] - exp2[i]) > 1e-4) {
				lout << "[Warning] i="<<i<<" "<< exp1[i] << " " << exp2[i] << " " << exp1[i] - exp2[i] << endl;
			}
		}
	}

	void ModelExam::TestHiddenExp(int nLen)
	{
		lout << "[ModelExam] Test Hidden Exp (nlen=" << nLen << ")" << endl;

		Vec<int> wseq(nLen);
		wseq.Fill(0);

		pm->ExactNormalize(nLen);

		Vec<double> exp1(pm->GetParamNum());
		Vec<double> exp2(pm->GetParamNum());
		exp1.Fill(0);
		exp2.Fill(0);

		LogP mglogp1, mglogp2;

		pm->GetHiddenExp(wseq, exp1.GetBuf());
		mglogp1 = pm->GetLogProb(wseq, true);

		Seq seq(nLen, pm->m_hlayer, pm->m_hnode);
		seq.x.Set(wseq.GetBuf(), nLen, pm->GetVocab());
		Vec<int> hnodeSeq(nLen);
		trf::VecIter iter(hnodeSeq.GetBuf(), nLen, 0, pm->GetEncodeHiddenLimit() - 1);
		LogP logpSum = trf::LogP_zero;
		while (iter.Next()) {
			pm->DecodeHidden(hnodeSeq, seq.h);
			LogP lp = pm->GetLogProb(seq, true);
			pm->FeatCount(seq, exp2, trf::LogP2Prob(lp));
			logpSum = trf::Log_Sum(logpSum, lp);
		}
		exp2 /= trf::LogP2Prob(logpSum);
		mglogp2 = logpSum;


		lout_variable(mglogp1);
		lout_variable(mglogp2);

		for (int i = 0; i < pm->GetParamNum(); i++) {
			if (fabs(exp1[i] - exp2[i]) > 1e-4) {
				lout << "[Warning] i=" << i << " " << exp1[i] << " " << exp2[i] << " " << exp1[i] - exp2[i] << endl;
			}
		}
	}

	void ModelExam::TestSample(int nLen /* = -1 */)
	{
		lout << "[ModelExam] Test Sample (nlen=" << nLen << ")" << endl;

		Vec<double> exp1(pm->GetParamNum());
		Vec<double> exp2(pm->GetParamNum());
		Vec<double> exp3(pm->GetParamNum());
		Vec<Prob> len1(pm->GetMaxLen() + 1);
		Vec<Prob> len2(pm->GetMaxLen() + 1);

		len1.Fill(1.0 / pm->GetMaxLen());
		pm->SetPi(len1.GetBuf());

		if (nLen > 0) {
			pm->ExactNormalize(nLen);
			pm->GetNodeExp(nLen, exp1.GetBuf());
		}
		else {
			pm->ExactNormalize();
			pm->GetNodeExp(exp1.GetBuf());
		}

		exp2.Fill(0);
		exp3.Fill(0);
		len2.Fill(0);
		File fdbg("test_sample_exp.dbg", "wt");
		fdbg.PrintArray("%f ", exp1.GetBuf(), exp1.GetSize());
		int nSampleNum = 1000;
		int nSampleTime = 100;
		Seq seq;
		pm->RandSeq(seq, nLen);
		for (int t = 1; t <= nSampleTime; t++) {
			for (int i = 0; i < nSampleNum; i++) {
				if (nLen > 0)
					pm->MarkovMove(seq);
				else
					pm->Sample(seq);

				pm->FeatCount(seq, exp2);
				pm->GetHiddenExp(VecShell<VocabID>(seq.wseq(), seq.GetLen()), exp3.GetBuf());
				len2[seq.GetLen()]++;
			}

			exp2 /= t * nSampleNum;
			exp3 /= t* nSampleNum;
			fdbg.PrintArray("%f ", exp2.GetBuf(), exp2.GetSize());
			fdbg.PrintArray("%f ", exp3.GetBuf(), exp3.GetSize());
			double diff2 = VecDiff(exp1.GetBuf(), exp2.GetBuf(), exp2.GetSize());
			double diff3 = VecDiff(exp1.GetBuf(), exp3.GetBuf(), exp3.GetSize());
			lout << "t=" << t << " diff2-1=" << diff2 << " diff3-1=" << diff3 << endl;
			exp2 *= t * nSampleNum;
			exp3 *= t * nSampleNum;
		}

		exp2 /= nSampleTime * nSampleNum;
		len2 /= nSampleTime * nSampleNum;
		for (int i = 1; i <= pm->GetMaxLen(); i++) {
			lout << i << "\t" << len1[i] << "\t" << len2[i] << endl;
		}
	}

	void SAExam::TestGradient()
	{
		ModelExam m(pfunc->GetModel());
		m.SetValueRand();
		m.pm->ExactNormalize();

		Vec<double> grad1(pfunc->GetParamNum());
		Vec<double> grad2(pfunc->GetParamNum());

	}
}