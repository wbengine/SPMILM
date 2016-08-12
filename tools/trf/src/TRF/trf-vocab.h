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
#include "trf-def.h"

namespace trf
{
	typedef int VocabID;
	const int VocabID_none = -1;
	const int VocabID_seqbeg = -3;
	const int VocabID_seqend = -2;
	static const char* Word_beg = "<s>";
	static const char* Word_end = "</s>";

	/**
	* \class
	* \brief HRF vocabulary, including <s> and </s>
	*/
	class Vocab
	{
	public:
		Array<VocabID> m_aWordID; ///< the word id. i.e 0,1,2,3,...
		Array<String> m_aWords; ///< the string of each vocabulary id
		Array<VocabID> m_aClass; ///< store the classes of each word. Support soft and hard class
 		Array<Array<VocabID>*> m_aClass2Word; ///< store the word belonging to each class.

	public:
		Vocab();
		Vocab(const char* pathVocab);
		~Vocab();
		/// get the vocab size, i.e. the word number
		int GetSize() { return m_aWords.GetNum(); }
		/// get the total class number
		int GetClassNum() { return m_aClass2Word.GetNum(); }
		/// get word string
		const char* GetWordStr(int id) {
			switch (id) {
			case VocabID_seqbeg: return Word_beg; break;
			case VocabID_seqend: return Word_end; break;
			default: return m_aWords[id].GetBuffer();
			}
			return NULL;
		}
		/// get class map
		VocabID *GetClassMap() { return m_aClass.GetBuffer(); }
		/// get class
		VocabID GetClass(VocabID wid) {
			if (wid >= m_aClass.GetNum())
				return VocabID_none;
			return m_aClass[wid];
		}
		/// get classes of a word sequence
		void GetClass(VocabID *pcid, const VocabID *pwid, int nlen);
		/// random a class
		VocabID RandClass() {
			if (GetClassNum() == 0)
				return VocabID_none;
			return rand() % GetClassNum();
		}
		/// get word belonging to a class
		Array<int> *GetWord(VocabID cid) {
			if (cid == VocabID_none) // if no class, then return the word id.
				return &m_aWordID;
			return m_aClass2Word[cid];
		}
		/// iter all the words, regardless the beg/end symbols
		int IterBeg() const { return 0; }
		/// iter all the words, regardless the beg/end symbols
		int IterEnd() const { return m_aWords.GetNum() - 1; }
		/// Check if the VocabID is a legal word
		bool IsLegalWord(VocabID id) const { return (id >= IterBeg() && id <= IterEnd()); }
	};
}
