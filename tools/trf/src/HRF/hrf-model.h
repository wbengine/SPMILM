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

#ifndef _HRF_MODEL_H_
#define _HRF_MODEL_H_

#include "trf-model.h"

namespace hrf
{
	typedef trf::PValue PValue; // parameter values
	typedef float HValue; // hidden variable valuse
	typedef trf::Prob Prob;
	typedef trf::LogP LogP;
	typedef trf::Vocab Vocab;
	typedef trf::VocabID VocabID;


	class Model;
	class AlgNode;
	class AlgLayer;

	/**
	* \class
	* \brief define the sequence including the sequence and the hidden variables
	*/
	class Seq
	{
	public:
		trf::Seq x;
		Mat<HValue> h; /// mutiple hidden matrix  [position * (layer * hnode)]
		int m_nLen; ///< current length
		int m_hlayer;
		int m_hnode;

	public:
		Seq() : m_nLen(0),m_hnode(0),m_hlayer(0) {};
		Seq(int len, int hlayer, int hnode) { Reset(len, hlayer, hnode); }
		Seq(Seq &seq) { Copy(seq); }
		void SetLen(int len) { m_nLen = len; }
		int GetLen() const { return m_nLen; }
		int GetHlayer() const { return m_hlayer; }
		int GetHnode() const { return m_hnode; }
		VecShell<VocabID> GetWordSeq() { return VecShell<VocabID>(x.GetWordSeq(), m_nLen); }
		VocabID *wseq() { return x.GetWordSeq(); }
		VocabID *cseq() { return x.GetClassSeq(); }
		void Reset(int len, int hlayer, int hnode);
		void Copy(Seq &seq);
		/// Return the sub-sequence
		Seq GetSubSeq(int nPos, int nOrder);
		/// If the two sequence is equal
		bool operator == (Seq &s);
		void Print();
		void Write(File &file);
	};

	/**
	* \class
	* \brief the forward-backward class for all the node( x and h )
	*/
	class AlgNode : public trf::Algfb
	{
	private:
		Model *m_pModel;
		Seq m_seq;
	public:
		AlgNode(Model *p);
		virtual LogP ClusterSum(int *pSeq, int nLen, int nPos, int nOrder);
	};

#define HHMap(h1, h2) (int)((h1) * 2 + (h2))

#define HRF_VALUE_SET(p, m) \
	memcpy(m.GetBuf(), p, sizeof(PValue)*m.GetSize()); \
	p += m.GetSize();
#define HRF_VALUE_GET(p, m) \
	memcpy(p, m.GetBuf(), sizeof(PValue)*m.GetSize()); \
	p += m.GetSize();


	/**
	 * \class Model
	 * \brief hidden-random-field model
	*/
	class Model : public trf::Model
	{
	public:
		int m_hlayer;			///< the number of hidden layer
		int m_hnode;			///< the number of hidden nodes
		Mat3d<PValue> m_m3dVH; ///< the weight between Word(V) and Hidden(H)
		Mat3d<PValue> m_m3dCH; ///< the weight between Class(C) and Hidden(H)
		Mat3d<PValue> m_m3dHH; ///< the weight between adjacent Hidden(H)
		Mat<PValue>   m_matBias; ///< the bias for each value of Hidden(H)

	protected:
		AlgNode m_nodeCal; ///< the forward-backward calculation for node (x and h)

	public:
		Model(Vocab *pv) :trf::Model(pv), m_hlayer(0), m_hnode(0), m_nodeCal(this),
			m_nSampleHAccTimes(0), m_nSampleHTotalTimes(0),
			m_nLenJumpAccTimes(0), m_nLenJumpTotalTime(0)
		{}
		Model(Vocab *pv, int hlayer, int hnode, int maxlen) : trf::Model(pv, maxlen), 
			m_hlayer(hlayer), m_hnode(hnode), m_nodeCal(this),
			m_nSampleHAccTimes(0), m_nSampleHTotalTimes(0),
			m_nLenJumpAccTimes(0), m_nLenJumpTotalTime(0)
		{
			Reset(pv, hlayer, hnode, maxlen);
		}
		/// reset, the maxlen is the length excluding the beg/end symbols.
		void Reset(Vocab *pv, int hlayer, int hnode, int maxlen);
		/// get hidden node dimension
		int GetHnode() const { return m_hnode; }
		/// Get HH mat order
		int GetHiddenOrder() const { return 2; }
		/// Get the total parameter number
		int GetParamNum() const { return trf::Model::GetParamNum() + m_m3dVH.GetSize() + m_m3dCH.GetSize() + m_m3dHH.GetSize() + m_matBias.GetSize(); }
		/// Set the parameters
		virtual void SetParam(PValue *pParam);
		/// Get the paremetre vector
		void GetParam(PValue *pParam);
		/// calculate the probability
		LogP GetLogProb(Seq &seq, bool bNorm = true);

		/// Read Model
		void ReadT(const char *pfilename);
		/// Write Model
		void WriteT(const char *pfilename);

	public:
		/* Exact calculatation function */
		/// [exact] calculate the probability of x
		LogP GetLogProb(VecShell<VocabID> &x, bool bNorm = true);
		/// [exact] Calculate the logP in each cluster. Only used for forward-backword algorithms ( class AlgNode)
		LogP ClusterSum(Seq &seq, int nPos, int nOrder);
		/// [exact] Calculate the logp in each cluster. Only consinder the VH,CH,HH values, used in class AlgHidden
		LogP HiddenClusterSum(Seq &seq, int nPos, int nOrder);
		/// [exact] Calculate the logp in each cluster. Only consinder the feature values
		LogP FeatClusterSum(trf::Seq &x, int nPos, int nOrder);
		/// [exact] Calculate the logp in each cluster. Only consinder the VH,CH,HH values on such layer
		LogP LayerClusterSum(Seq &seq, int nlayer, int nPos, int nOrder);
		/// [exact] Exact Normalization, return the logz of given length
		double ExactNormalize(int nLen);
		/// [exact] Exact Normalization all the length
		void ExactNormalize();
		/// [exact] Exactly calculate the marginal probability at position 'nPos' and with order 'nOrder'
		LogP GetMarginalLogProb(int nLen, int nPos, Seq &sub, bool bNorm = true);
		/// [exact] sum_l { n_l/n * E_{p_l}[f] }: Exactly calculate the expectation over x and h
		void GetNodeExp(double *pExp, Prob *pLenProb = NULL);
		/// [exact] E_{p_l}[f]: Exactly calculate the expectation over x and h for length nLen
		void GetNodeExp(int nLen, double *pExp);
		/// [exact] E_{p_l}[f]: Exactly calculate the expectation over x and h for length nLen
		void GetNodeExp(int nLen, VecShell<double> featexp,
			Mat3dShell<double> VHexp, Mat3dShell<double> CHexp, Mat3dShell<double> HHexp,
			MatShell<double> Bexp);
		/// [exact] E_{p_l(h|x)}[f]: don't clean the pExp and directly add the new exp to pExp.
		void GetHiddenExp(VecShell<int> x, double *pExp);
		/// [exact] called in GetHiddenExp.
		void GetLayerExp(AlgLayer &fb, int nLayer,
			Mat3dShell<double> &VHexp, Mat3dShell<double> &CHexp, Mat3dShell<double> &HHexp, MatShell<double> &Bexp,
			LogP logz = 0);

	public:
		int m_nSampleHAccTimes; ///< sample H the acceptance times
		int m_nSampleHTotalTimes; ///< sample H the total times
		int m_nLenJumpAccTimes; ///< lenght jump the acceptance times
		int m_nLenJumpTotalTime; ///< total times of length jump
		/// [sample] Perform one train-dimensional mixture sampling
		void Sample(Seq &seq);
		/// [sample] Local Jump - sample a new length
		void LocalJump(Seq &seq);
		/// [sample] Markov Move - perform the gibbs sampling
		void MarkovMove(Seq &seq);
		/// [sample] Propose the length, using the variable m_matLenJump
		LogP ProposeLength(int nOld, int &nNew, bool bSample);
		/// [sample] Propose the h_{i} at position i. Then return the propose probability Q(h_i|h_{other})
		LogP ProposeH0(VecShell<HValue> &hi, Seq &seq, int nPos, bool bSample);
		/// [sample] Propose the c_{i} at position i. Then return the propose probability R(c_i|h_i,c_{other})
		LogP ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample);
		/// [sample] A reduced model only consinder HHmat(W) and VHmat(M) and CHmat(U).
		/*
		* \param [out] logps return K values, denoting logP(h_{i,k} = 1), k=1,...,K
		* \param [in] seq the sequence.
		* \param [in] nPos position i.
		* \param [in] bConsiderXandC =true mean considering the VH and CH matrix, otherwise only HH matrix.
		* \details The function return the probability of each component and used to sample h_i or calculate Q.
		*/
		void ProposeHProbs(VecShell<LogP> &logps, Seq &seq, int nPos, bool bConsiderXandC = false);
		/// [sample] Return the distribution of c_i at position nPos
		void ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model. It only consindering the HH matrix (W)
		LogP GetReducedModelForH(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample class c_i, consindering CH matrix(U) and class-ngram (lambda_c)
		LogP GetReducedModelForC(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample word w_i, consindering VH matrix(M) and word-ngram (lambda_w)
		LogP GetReducedModelForW(Seq &seq, int nPos);
		/// [sample] using the logprobs returned by ProposeHProb to calculate the logprob of hi.
		LogP GetConditionalProbForH(VecShell<HValue> &hi, VecShell<Prob> &probs);
		/// [sample] Fixed h, given c_i, summate the probabilities of x_i, i.e. P(c_i)
		LogP GetMarginalProbOfC(Seq &seq, int nPos);
		/// [sample] Sample the c_i at position nPos given h_i without x_i.
		void SampleC(Seq &seq, int nPos);
		/// [sample] Sample the w_i at position nPos
		/* if bSample=ture, then sample w[nPos]. Otherwise only calculate the conditional probabilities of current w[nPos]. */
		LogP SampleW(Seq &seq, int nPos, bool bSample = true);
		/// Random init sequence, if nLen==-1, random the length also.
		void RandSeq(Seq &seq, int nLen = -1);
		/// Random init the hidden variables
		void RandHidden(Seq &seq);
		/// [sample] sample h given x using gibbs sampling.
		/* different with SampleH, which using MH sample to sample h
		If tagH != NULL, then just calcualte the transition probability.
		*/
		virtual LogP SampleHAndCGivenX(Seq &seq, MatShell<HValue> *tagH = NULL);
		
	public:
		/// encode the x_i and h_i at position i to a integer
		int EncodeNode(VocabID xi, VocabID ci, VecShell<HValue> &hi);
		/// encode the x and h to a integer sequence
		void EncodeNode(VecShell<int> &vn, Seq &seq, int nPos = 0, int nDim = -1);
		/// decode a integer to the x_i and h_i
		void DecodeNode(int n, VocabID &xi, VocabID &ci, VecShell<HValue> &hi);
		/// decode several integer to a sequence
		void DecodeNode(VecShell<int> &vn, Seq &seq, int nPos = 0, int nDim = -1);
		/// The encoded integer size
		int GetEncodeNodeLimit() const;
		/// encode the hidden vector h_i to a integer
		int EncodeHidden(VecShell<HValue> hi);
		/// decode a integer to a hidden vector
		void DecodeHidden(int n, VecShell<HValue> hi);
		/// decoder several integer to a sequence
		void DecodeHidden(VecShell<int> &vn, Mat<HValue> &h, int nPos = 0, int nDim = -1);
		/// The encoded integer size
		int GetEncodeHiddenLimit() const;
		/// decoder several integer to a sequence
		void DecodeLayer(VecShell<int> &vn, Mat<HValue> &h, int layer, int nPos = 0, int nDim = -1);
		/// The encoded integer size of one layer
		int GetEncodeLayerLimit() const;

	public:
		/// Map a paremeter vector to each kinds of parameters
		template <typename T>
		void BufMap(T *p, VecShell<T> &feat, Mat3dShell<T> &VH, Mat3dShell<T> &CH, Mat3dShell<T> &HH, MatShell<T> &Bias)
		{
			feat.Reset(p, trf::Model::GetParamNum());
			p += feat.GetSize();

			VH.Reset(p, m_m3dVH.GetXDim(), m_m3dVH.GetYDim(), m_m3dVH.GetZDim());
			p += VH.GetSize();
			CH.Reset(p, m_m3dCH.GetXDim(), m_m3dCH.GetYDim(), m_m3dCH.GetZDim());
			p += CH.GetSize();
			HH.Reset(p, m_m3dHH.GetXDim(), m_m3dHH.GetYDim(), m_m3dHH.GetZDim());
			p += HH.GetSize();
			Bias.Reset(p, m_matBias.GetRow(), m_matBias.GetCol());
		}
		/// Count the feature number in current sequence, and add to the result
		void FeatCount(Seq &seq, VecShell<double> featcount,
			Mat3dShell<double> VHcount, Mat3dShell<double> CHcount, Mat3dShell<double> HHcount, 
			MatShell<double> Bcount,
			double dadd = 1);
		/// Count the hidden features
		void HiddenFeatCount(Seq &seq,
			Mat3dShell<double> VHcount, Mat3dShell<double> CHcount, Mat3dShell<double> HHcount,
			MatShell<double> Bcount,
			double dadd = 1);
		/// Count the feature number in current sequence
		void FeatCount(Seq &seq, VecShell<double> count, double dadd = 1);

	public:
		PValue SumVHWeight(MatShell<PValue> m, VecShell<HValue> h);
		PValue SumHHWeight(Mat3dShell<PValue> m, VecShell<HValue> h1, VecShell<HValue> h2);
		PValue SumVHWeight(MatShell<PValue> m, VecShell<HValue> h, int layer);
		PValue SumHHWeight(Mat3dShell<PValue> m, VecShell<HValue> h1, VecShell<HValue> h2, int layer);
	};


	/**
	* \class
	* \brief the forward-backward class for the hidden variables (h)
	*/
	class AlgLayer : public trf::Algfb
	{
	public:
		Model *m_pModel;
		Seq m_seq;
		int m_nlayer; ///< the layer
	public:
		AlgLayer(Model *p, VecShell<VocabID> x, int nlayer);
		virtual LogP ClusterSum(int *pSeq, int nLen, int nPos, int nOrder);
	};

}


#endif