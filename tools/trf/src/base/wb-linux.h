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
#ifdef __linux
#include <iostream>
#include <ctime>
#include <cstring>
using namespace std;
#include <stdio.h>
#include <stdlib.h>

#define _ftelli64 ftello64
#define _fseeki64 fseeko64
#define _vsnprintf vsnprintf

namespace wb
{
	void _strdate(char *str);
	void _strtime(char *str);
	void getch();
}
#endif //__linux
