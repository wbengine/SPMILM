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
// All h, cpp, cc, cu, cuh and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.


#include "cutrf-def.cuh"
#include "trf-vocab.h"

namespace cutrf
{
	typedef trf::VocabID VocabID;
	const int VocabID_none = trf::VocabID_none;

	/**
	* \class
	* \brief HRF vocabulary, including <s> and </s>
	*/
	class Vocab
	{
	private:
		cu::Vec<VocabID> m_aWordID; ///< the word id. i.e 0,1,2,3,...
		cu::Vec<VocabID> m_aClass; ///< store the classes of each word. Support soft and hard class

 		cu::Vec<cu::Reg> m_aClass2WordReg; ///< store the word belonging to each class.
		cu::Vec<VocabID> m_aClass2WordBuf;
		int m_nMaxWordInOneClass;

	public:
		__host__ Vocab() {};
		__host__ Vocab(trf::Vocab &v) { Copy(v); }
		/// Release the vocab
		__host__ void Release()
		{
			m_aWordID.Reset();
			m_aClass.Reset();
		
			m_aClass2WordReg.Reset();
			m_aClass2WordBuf.Reset();
		}
		/// Copy to device
		__host__ void Copy(trf::Vocab &v)
		{
			m_aWordID.Copy(v.m_aWordID);
			m_aClass.Copy(v.m_aClass);

			if (v.GetClassNum() == 0) {
				m_nMaxWordInOneClass = v.GetSize();
			}
			else {
				/* copy the Class2Word list */
				int nTotalBuf = 0;
				m_nMaxWordInOneClass = 0;
				wb::Array<cu::Reg> aReg;
				wb::Array<VocabID> aBuf;
				for (int i = 0; i < v.m_aClass2Word.GetNum(); i++) {
					cu::Reg reg;
					reg.nBeg = nTotalBuf;
					reg.nEnd = nTotalBuf + v.m_aClass2Word[i]->GetNum();
					aReg.Add(reg);
					aBuf.Add(*v.m_aClass2Word[i]);
					nTotalBuf += v.m_aClass2Word[i]->GetNum();
					m_nMaxWordInOneClass = max(m_nMaxWordInOneClass, reg.nEnd - reg.nBeg);
				}
				lout_assert(nTotalBuf == aBuf.GetNum());
				m_aClass2WordReg.Copy(aReg);
				m_aClass2WordBuf.Copy(aBuf);
				lout << "Class2Word Buf=" << nTotalBuf << endl;
			}

			lout << "MaxWordInOneClass=" << m_nMaxWordInOneClass << endl;
		}
		__host__ int ByteSize() {
			return m_aWordID.ByteSize() + m_aClass.ByteSize() + m_aClass2WordReg.ByteSize() + m_aClass2WordBuf.ByteSize();
		}
		/// get the vocab size, i.e. the word number
		__host__ __device__ int GetSize() { return m_aWordID.GetSize(); }
		/// get the total class number
		__host__ __device__ int GetClassNum() { return m_aClass2WordReg.GetSize(); }
		/// get the maximum number of word in one class
		__host__ __device__ int GetMaxWordInOneClass() const { return m_nMaxWordInOneClass; }
		/// get class map
		__device__ VocabID *GetClassMap() { return m_aClass.GetBuf(); }
		/// get class
		__device__ VocabID GetClass(VocabID wid) {
			if (wid >= m_aClass.GetSize())
				return VocabID_none;
			return m_aClass[wid];
		}
		/// get classes of a word sequence
		__device__ void GetClass(VocabID *pcid, const VocabID *pwid, int nlen)
		{
			for (int i = 0; i < nlen; i++) {
				pcid[i] = GetClass(pwid[i]);
			}
		}
		/// get word belonging to a class
		__device__ cu::VecShell<VocabID> GetWord(VocabID cid) {
			if (cid == VocabID_none) // if no class, then return the word id.
				return cu::VecShell<VocabID>(m_aWordID.GetBuf(), m_aWordID.GetSize());

			cu::Reg reg = m_aClass2WordReg[cid];
			return cu::VecShell<VocabID>(m_aClass2WordBuf.GetBuf() + reg.nBeg, reg.nEnd - reg.nBeg);
		}
		/// iter all the words, regardless the beg/end symbols
		__device__ int IterBeg() const { return 0; }
		/// iter all the words, regardless the beg/end symbols
		__device__ int IterEnd() const { return m_aWordID.GetSize() - 1; }
		/// Check if the VocabID is a legal word
		__device__ bool IsLegalWord(VocabID id) const { return (id >= IterBeg() && id <= IterEnd()); }

		/// get a random class
		__device__ VocabID RandClass(RandState *state) {
			if (GetClassNum() == 0)
				return VocabID_none;
			return curand(state) % GetClassNum();
		}
	};
}