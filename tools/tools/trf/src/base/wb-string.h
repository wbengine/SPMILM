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


/**
* \file
* \author WangBin
* \date 2016-05-05
* \brief define the class String
*/



#pragma once
#include <cstring>
#include <algorithm>
#include <cstdarg>	//include va_list
#include <string.h>
using namespace std;
#include "wb-vector.h"
#include "wb-linux.h"


namespace wb
{

	const int cn_default_str_len = 10;  ///< default length of string
	const int cn_default_max_len = 32 * 1024;  ///< default maximum length 

	/// a dynamic string class
	/*
		A lot of function for string operation, including:
		- operation "=";
		- compare operation >/</== and so on
		- other operations.
	*/
	class String
	{
	private:
		char *m_pBuffer;  ///< string buffer
		int m_nBufSize;   ///< string buffer size. Note this is not the string length, just the buffer size.

		char *m_tokPtr; /// store the top pointer. used in 
	public:
		String(int p_nLen = cn_default_str_len);
		String(const char *p_str);
		String(const char* p_str, int nLen);
		String(char c);
		String(const String &p_str);
		~String();
		/// Reset the string leng
		/* If the nLen<=m_nBufSize, then do nothing. \n
		   If the nLen>m_nBufSize, then re-alloc memory.
		   Return the buffer pointer.
		*/
		char* Reset(int nLen);
		/// get buffer
		char *GetBuffer() const { return m_pBuffer; }
		/// get the pointer to the last position
		char *End() const { return m_pBuffer + strlen(m_pBuffer) - 1; } 
		/// get buffer size
		int GetSize() const { return m_nBufSize; }
		/// get string length
		int GetLength() const { return strlen(m_pBuffer); }
		/// set the string = "", but donot release the buffer
		void Clean() { m_pBuffer[0] = '\0'; }
		/// format print to string
		const char* Format(const char* p_pMessage, ...);
		/// operator =
		void operator = (const String &p_str);
		/// operator =
		void operator = (const char *pStr);
		/// operator (char*)
		operator char*() const { return m_pBuffer; }
		/// operator []
		char &operator[] (int i) const { return m_pBuffer[i]; }
		//@{
		/// compare function
		bool operator >(const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) > 0; }
		bool operator < (const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) < 0; }
		bool operator >= (const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) >= 0; }
		bool operator <= (const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) <= 0; }
		bool operator == (const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) == 0; }
		bool operator != (const String &p_str) { return strcmp(m_pBuffer, p_str.m_pBuffer) != 0; }
		//@}
		/// operator +=
		String operator += (const String &str) { strcat(Reset(strlen(m_pBuffer) + strlen(str) + 1), str.m_pBuffer); return *this; }
		//@{
		/// compare function
		bool operator > (const char *p) { return strcmp(m_pBuffer, p) > 0; }
		bool operator < (const char *p) { return strcmp(m_pBuffer, p) < 0; }
		bool operator >= (const char *p) { return strcmp(m_pBuffer, p) >= 0; }
		bool operator <= (const char *p) { return strcmp(m_pBuffer, p) <= 0; }
		bool operator == (const char *p) { return strcmp(m_pBuffer, p) == 0; }
		bool operator != (const char *p) { return strcmp(m_pBuffer, p) != 0; }
		//@}
		/// operator +=
		String operator += (const char *p) { strcat(Reset(strlen(m_pBuffer) + strlen(p) + 1), p); return *this; }

		friend String operator + (const String &str, const char *p) {
			return String(str) += p;
		}
		friend String operator + (const char *p, const String &str) {
			return String(p) += str;
		}
		friend String operator + (const String &str1, const String &str2) {
			return String(str1) += str2;
		}
		/// to upper
		char *Toupper();
		/// to lower
		char *Tolower();
		/// delete a sub-string
		void DeleteSub(int nLocal, int nLen);
		/// Find
		int Find(const char *sub);
		/// replace
		String Replace(const char *src, const char *rpl);
		/// split begin
		inline char *TokBegin(const char *p);
		/// split next
		inline char *TokSub(const char *p);
		/// split to string array. Using strtok().
		void Split(Array<String> &aStrs, const char* delimiter);
		/// if the string is a path, this function return the file name.
		String FileName();
	};
}


