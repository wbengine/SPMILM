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


#include "wb-string.h"

namespace wb
{
	String::String(int p_nLen /*= cn_default_str_len*/) :m_pBuffer(NULL), m_nBufSize(0), m_tokPtr(NULL)
	{ 
		Reset(p_nLen+1);
		m_pBuffer[0] = '\0';
	}
	String::String(const char *p_str) : m_pBuffer(NULL), m_nBufSize(0), m_tokPtr(NULL)
	{
		int nlen = strlen(p_str);
		Reset(nlen + 1);
		memcpy(m_pBuffer, p_str, nlen + 1);
	}
	String::String(const char* p_str, int nLen) :m_pBuffer(NULL), m_nBufSize(0), m_tokPtr(NULL)
	{
		Reset(nLen + 1);
		nLen = min(nLen, (int)strlen(p_str));
		memcpy(m_pBuffer, p_str, nLen);
		m_pBuffer[nLen] = '\0';
	}
	String::String(char c) :m_pBuffer(NULL), m_nBufSize(0), m_tokPtr(NULL)
	{
		Reset(2);
		m_pBuffer[0] = c;
		m_pBuffer[1] = '\0';
	}
	String::String(const String &p_str) :m_pBuffer(NULL), m_nBufSize(0), m_tokPtr(NULL)
	{
		Reset(p_str.m_nBufSize);
		memcpy(m_pBuffer, p_str.m_pBuffer, m_nBufSize);
	}
	String::~String()
	{
		delete[]m_pBuffer;
		m_nBufSize = 0;
	}
	char* String::Reset(int nLen)
	{
		if (nLen > m_nBufSize) {
			char *pNew = new char[nLen];
			if (m_pBuffer) {
				memcpy(pNew, m_pBuffer, sizeof(char)*m_nBufSize);
				delete[] m_pBuffer;
			}
			m_pBuffer = pNew;
			m_nBufSize = nLen;
		}
		return m_pBuffer;
	}
	const char* String::Format(const char* p_pMessage, ...)
	{
		char strBuffer[cn_default_max_len];
		va_list vaParams;
		va_start(vaParams, p_pMessage);

		_vsnprintf(strBuffer, cn_default_max_len, p_pMessage, vaParams);

		int nLen = strlen(strBuffer);
		Reset(nLen + 1);
		memcpy(m_pBuffer, strBuffer, sizeof(char)*(nLen + 1));

		return m_pBuffer;
	}
	void String::operator = (const String &p_str)
	{
		Reset(p_str.m_nBufSize);
		memcpy(m_pBuffer, p_str.m_pBuffer, p_str.m_nBufSize);
	}

	void String::operator = (const char *pStr)
	{
		int nLen = strlen(pStr);
		Reset(nLen + 1);
		memcpy(m_pBuffer, pStr, nLen + 1);
	}
	void String::DeleteSub(int nLocal, int nLen)
	{
		m_pBuffer[nLocal] = '\0';
		*this += m_pBuffer + nLocal + nLen;
	}
	int String::Find(const char *sub)
	{
		char *p = strstr(m_pBuffer, sub);
		if (p) {
			return p - m_pBuffer;
		}
		return -1;
	}
	String String::Replace(const char *src, const char *rpl)
	{
		int nLocal = Find(src);
		if (nLocal == -1) {
			return "";
		}

		String strnew(m_pBuffer, nLocal);
		return strnew + rpl + &m_pBuffer[nLocal + strlen(src)];
	}
	char *String::TokBegin(const char *p)
	{
		m_tokPtr = m_pBuffer;
		return TokSub(p);
	}
	char *String::TokSub(const char *p)
	{
		if (*m_tokPtr == '\0')
			return NULL;

		char *pBeg = NULL;
		for (; *m_tokPtr != '\0'; m_tokPtr++)
		{
			if (pBeg == NULL) {
				//�ҵ���ʼ��
				if (NULL == strchr(p, *m_tokPtr)) {
					pBeg = m_tokPtr;
				}
			}
			else {
				//�ҵ�������
				if (strchr(p, *m_tokPtr)) {
					*m_tokPtr = '\0';
					m_tokPtr++;
					break;
				}
			}
		}

		return pBeg;
	}
	void String::Split(Array<String> &aStrs, const char* delimiter)
	{
		char *psave = strdup(m_pBuffer); // save the string

		aStrs.Clean();
		char *p = strtok(psave, delimiter);
		while (p) {
			aStrs.Add(p);
			p = strtok(NULL, delimiter);
		}

		free(psave); // free the string
	}
	String String::FileName()
	{
		char *p = strrchr(m_pBuffer, '.');
		if (p)
			return String(m_pBuffer, p - m_pBuffer);
		return String(m_pBuffer);
	}
	char *String::Toupper()
	{
		char *p = m_pBuffer;
		while (*p = toupper(*p))
			p++;
		return m_pBuffer;
	}
	char *String::Tolower()
	{
		char *p = m_pBuffer;
		while (*p = tolower(*p))
			p++;
		return m_pBuffer;
	}
}


