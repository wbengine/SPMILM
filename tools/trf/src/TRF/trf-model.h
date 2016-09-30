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
#include "trf-feature.h"
#include "trf-alg.h"
#include <omp.h>

namespace trf
{


	class Model;

	/**
	 * \class AlgNode
	 * \brief the forward-backward algorithms for TRF model
	 */
	class AlgNode : public Algfb
	{
	private:
		Model *m_pModel;
		Seq m_seq;
	public:
		AlgNode(Model *p) :m_pModel(p){};
		virtual LogP ClusterSum(int *pSeq, int nLen, int nPos, int nOrder);
	};
	/**
	 * \class Model
	 * \brief TRF model
	 */
	class Model
	{
	public:
		Feat *m_pFeat;	 ///< hash all the features
		Vec<PValue> m_value; ///< the value for each features

		int m_maxlen; ///< the maximum length of model, excluding <s> and </s>. The min-len = 1
		Vec<Prob> m_pi; ///< the prior length distribution \pi
		Vec<LogP> m_logz; ///< the normalization constants log Z_l
		Vec<LogP> m_zeta; ///< the estimated normalization constants \zeta_l (fix \zeta_1 = 0)

		Vocab *m_pVocab;

		Mat<Prob> m_matLenJump; ///< [sample] used to propose a new length
		int m_maxSampleLen; ///< [sample] the maximum sample length, default = m_maxlen + 2

	protected:
		AlgNode m_AlgNode; ///< the  forward-backward calculation each node

	public:
		/// constructor
		Model(Vocab *pv) :
			m_pFeat(NULL), 
			m_maxlen(0), 
			m_pVocab(pv), 
			m_AlgNode(this){
			m_nLenJumpAccTimes = 0;
			m_nLenJumpTotalTime = 0;
			m_nSampleHAccTimes = 0;
			m_nSampleHTotalTimes = 0;
		};
		/// constructor
		Model(Vocab *pv, int maxlen) :
			m_pFeat(NULL),
			m_maxlen(0),
			m_pVocab(pv),
			m_AlgNode(this) {
			Reset(pv, maxlen);
			m_nLenJumpAccTimes = 0;
			m_nLenJumpTotalTime = 0;
			m_nSampleHAccTimes = 0;
			m_nSampleHTotalTimes = 0;
		}
		/// destructor
		~Model()
		{
			SAFE_DELETE(m_pFeat);
		}
		/// Get max-len
		int GetMaxLen() const { return m_maxlen; }
		/// Get Vocab
		Vocab *GetVocab() const { return m_pVocab; }
		/// Get maximum order
		int GetMaxOrder() const { return m_pFeat->GetMaxOrder(); }
		/// Get parameter number
		int GetParamNum() const { return m_pFeat->GetNum(); }
		/// reset, the maxlen is the length excluding the beg/end symbols.
		void Reset(Vocab *pv, int maxlen);
		/// Set the parameters
		virtual void SetParam(PValue *pValue);
		/// Get the paremetre vector
		void GetParam(PValue *pValue);
		/// Set the pi
		void SetPi(Prob *pPi);
		/// Set updated zeta
		template <typename T>
		void SetZeta(T *pzeta)
		{
			ExactNormalize(1);
			for (int i = 1; i <= m_maxlen; i++) {
				m_zeta[i] = (LogP)( pzeta[i] - pzeta[1] );
				m_logz[i] = (LogP)( m_zeta[i] + m_logz[1] );
			}
		}
		/// calculate the probability
		LogP GetLogProb(Seq &seq, bool bNorm = true);
		/// load ngram features from corpus
		void LoadFromCorpus(const char *pcorpus, const char *pfeatstyle, int nOrder);
		/// Count the feature number in a sequence
		void FeatCount(Seq &seq, double *pCount, double dadd = 1.0);

		/// Read Model
		void ReadT(const char *pfilename);
		/// Write Model
		void WriteT(const char *pfilename);
		/// Read Binary
// 		void ReadB(const char *pfilename);
// 		/// Write Binary
// 		void WriteB(const char *pfilename);

		/************************************************************************/
		/*exactly calculation functions                                         */
		/************************************************************************/
	public:
		/// [exact] Calculate the logP in each cluster. Only used for forward-backword algorithms ( class AlgNode)
		LogP ClusterSum(Seq &seq, int nPos, int nOrder);
		/// [exact] Exact Normalization, return the logz of given length
		double ExactNormalize(int nLen);
		/// [exact] Exact Normalization
		void ExactNormalize();
		/// [exact] E_{p_l}[f]: Exactly calculate the expectation over x and h for length nLen
		void GetNodeExp(int nLen, double *pExp);
		/// [exact] sum_l { n_l/n * E_{p_l}[f] }: Exactly calculate the expectation over x and h
		void GetNodeExp(double *pExp, Prob *pLenProb = NULL);

		/************************************************************************/
		/*sampling functions                                                    */
		/************************************************************************/
	public:
		int m_nLenJumpAccTimes; ///< lenght jump the acceptance times
		int m_nLenJumpTotalTime; ///< total times of length jump
		int m_nSampleHAccTimes; ///< sample H the acceptance times
		int m_nSampleHTotalTimes; ///< sample H the total times
		/// [sample] Perform one train-dimensional mixture sampling
		void Sample(Seq &seq);
		/// [sample] Local Jump - sample a new length
		void LocalJump(Seq &seq);
		/// [sample] Markov Move - perform the gibbs sampling
		virtual void MarkovMove(Seq &seq);
		/// [sample] Propose the length, using the variable m_matLenJump
		LogP ProposeLength(int nOld, int &nNew, bool bSample);
		/// [sample] Propose the c_{i} at position i. Then return the propose probability R(c_i|h_i,c_{other})
		LogP ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample);
		/// [sample] Return the propose distribution of c_i at position nPos
		void ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample class c_i.
		/* To reduce the computation cost, we using the following function to replace GetLogProb
		   when sampling c_i at position i in function ProposeCProbs and ProposeC0.
		   There we only consinder the features depending on c_i and indenpending with w_i,
		   i.e. calculating the propose prob without knowing the w_i at position i.
		*/
		LogP GetReducedModelForC(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample word w_i.
		LogP GetReducedModelForW(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced depending on nPos.
		LogP GetReducedModel(Seq &seq, int nPos);
		/// [sample] given c_i, summate the probabilities of x_i, i.e. P(c_i)
		LogP GetMarginalProbOfC(Seq &seq, int nPos);
		/// [sample] Sample the c_i at position nPos without x_i.
		/* the only differnece with ProposeC0 is than 
		   SampleC will accept the current class after propose it.
		   While ProposeC0 not.
		   ProposeC0 used in local jump. It cannot accept the propose c0 as there is no intial value of c_i.
		   SampleC used in Markov move.
		*/
		void SampleC(Seq &seq, int nPos);
		/// [sample] Sample the x_i at position nPos
		/* if bSample=ture, then sample x[nPos]. Otherwise only calculate the conditional probabilities of current x[nPos]. */
		LogP SampleX(Seq &seq, int nPos, bool bSample = true);
	public:
		/// perform AIS to calculate the normalization constants, return the logz of given length
		LogP AISNormalize(int nLen, int nChain, int nInter);
		void AISNormalize(int nLenMin, int nLenMax, int nChain, int nInter);
	};

	/**
	 * \class Model_FastSample
	 * \brief TRF model, revise the sample method to speedup the MCMC
	 */
	class Model_FastSample : public Model
	{
	public:
		int m_nMHtimes;
	public:
		Model_FastSample(Vocab *pv) :Model(pv) {
			m_nMHtimes = 1;
		}
		Model_FastSample(Vocab *pv, int maxlen) :Model(pv, maxlen) {
			m_nMHtimes = 1;
		}
		LogP ProposeW0(VocabID &wi, Seq &seq, int nPos, bool bSample = true)
		{
			Array<VocabID> *pXs = m_pVocab->GetWord(seq.x[class_layer][nPos]);
			Array<LogP> aLogps;

			VocabID nSaveX = seq.x[word_layer][nPos]; // save w[nPos]
			for (int i = 0; i < pXs->GetNum(); i++) {
				seq.x[word_layer][nPos] = pXs->Get(i);

// 				LogP d = 0;
// 				Array<int> afeat;
// 				m_pFeat->Find(afeat, seq, nPos, 1);
// 				for (int i = 0; i < afeat.GetNum(); i++)
// 					d += m_value[afeat[i]];
				aLogps[i] = 1; //GetReducedModelForW(seq, nPos);
			}
			seq.x[word_layer][nPos] = nSaveX;
			LogLineNormalize(aLogps, pXs->GetNum());

			int idx;
			if (bSample) {
				/* sample a value for x[nPos] */
				idx = LogLineSampling(aLogps, pXs->GetNum());
				wi = pXs->Get(idx);
			}
			else {
				idx = pXs->Find(nSaveX); // find nSave in the array.
				if (idx == -1) {
					lout_error("Can't find the VocabID(" << nSaveX << ") in the array.\n"
						<< "This may beacuse word(" << nSaveX << ") doesnot belongs to class("
						<< seq.x[class_layer][nPos] << ")");
				}
			}
			
			return aLogps[idx];
		}

		void ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos)
		{
			VocabID savecid = seq.x[class_layer][nPos];
			for (int cid = 0; cid < m_pVocab->GetClassNum(); cid++) {
				seq.x[class_layer][nPos] = cid;
				logps[cid] = 1;
			}
			seq.x[class_layer][nPos] = savecid;
			LogLineNormalize(logps.GetBuf(), m_pVocab->GetClassNum());
		}

		void MarkovMove(Seq &seq)
		{
			for (int i = 0; i < seq.GetLen(); i++)
				SamplePos(seq, i);
		}
		void SamplePos(Seq &seq, int nPos)
		{
			for (int times = 0; times < m_nMHtimes; times++) 
			{

				VocabID old_c = seq.x[class_layer][nPos];
				VocabID old_w = seq.x[word_layer][nPos];
				LogP pold = GetReducedModel(seq, nPos);


				VocabID prop_c = omp_nrand(0, m_pVocab->GetClassNum());
				Array<VocabID> *pWords = m_pVocab->GetWord(prop_c);
				int prop_w_id = omp_nrand(0, pWords->GetNum());
				VocabID prop_w = pWords->Get(prop_w_id);

				seq.x[class_layer][nPos] = prop_c;
				seq.x[word_layer][nPos] = prop_w;
				LogP pnew = GetReducedModel(seq, nPos);

				LogP g_old = Prob2LogP(1.0 / m_pVocab->GetClassNum()) + Prob2LogP(1.0 / m_pVocab->GetWord(old_c)->GetNum());
				LogP g_new = Prob2LogP(1.0 / m_pVocab->GetClassNum()) + Prob2LogP(1.0 / m_pVocab->GetWord(prop_c)->GetNum());
				LogP acclogp = pnew + g_old - (pold + g_new);

				if (Acceptable(LogP2Prob(acclogp))) {
					m_nSampleHAccTimes++;
					seq.x[class_layer][nPos] = prop_c;
					seq.x[word_layer][nPos] = prop_w;
				}
				else {
					seq.x[class_layer][nPos] = old_c;
					seq.x[word_layer][nPos] = old_w;
				}
				m_nSampleHTotalTimes++;

				lout_assert(seq.x[class_layer][nPos] == m_pVocab->GetClass(seq.x[word_layer][nPos]));
			}

// 			Vec<LogP> vlogps_c(m_pVocab->GetClassNum());
// 			ProposeCProbs(vlogps_c, seq, nPos);
// 			VocabID ci = seq.x[class_layer][nPos];
// 			VocabID C0 = LogLineSampling(vlogps_c.GetBuf(), vlogps_c.GetSize());
// 			LogP gci = vlogps_c[ci];
// 			LogP gc0 = vlogps_c[C0];
// 
// 			VocabID wi = seq.x[word_layer][nPos];
// 			VocabID w0;
// 			seq.x[class_layer][nPos] = ci;
// 			LogP gwi_ci = ProposeW0(wi, seq, nPos, false);
// 			seq.x[class_layer][nPos] = C0;
// 			LogP gw0_c0 = ProposeW0(w0, seq, nPos, true);
// 
// 			seq.x[class_layer][nPos] = ci;
// 			seq.x[word_layer][nPos] = wi;
// 			LogP pold = GetReducedModel(seq, nPos);
// 			seq.x[class_layer][nPos] = C0;
// 			seq.x[word_layer][nPos] = w0;
// 			LogP pnew = GetReducedModel(seq, nPos);
// 			
// 			LogP acclogp = pnew + gci + gwi_ci - (pold + gc0 + gw0_c0);
// 			if (Acceptable(LogP2Prob(acclogp))) {
// 				m_nSampleHAccTimes++;
// 				seq.x[class_layer][nPos] = C0;
// 				seq.x[word_layer][nPos] = w0;
// 			}
// 			else {
// 				seq.x[class_layer][nPos] = ci;
// 				seq.x[word_layer][nPos] = wi;
// 			}
// 			m_nSampleHTotalTimes++;
		}
	};
}