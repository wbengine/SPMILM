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


#include "wb-linux.h"
#ifdef __linux
namespace wb
{
	const char *wday[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
	void _strdate(char *str)
	{
		time_t timep;
		struct tm *p;
		time(&timep);
		p = localtime(&timep);
		sprintf(str, "%d%d%d %s", (1900+p->tm_yday), (1+p->tm_mon), p->tm_mday, wday[p->tm_wday]);
	}
	void _strtime(char *str)
	{
		time_t timep;
		struct tm *p;
		time(&timep);
		p = localtime(&timep);
		sprintf(str, "%d:%d:%d", p->tm_hour, p->tm_min, p->tm_sec);
	}
	void getch()
	{
	}

}
#endif
