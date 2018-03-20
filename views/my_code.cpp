#include <bits/stdc++.h>
#define ll long long int
#define vec vector<ll>
#define pb(x) push_back(x)
#define mp(x,y) make_pair(x,y)
#define pq priority_queue<ll> 

using namespace std;

int main() {
	freopen("input.txt","r",stdin);  
	ios::sync_with_stdio(false);
	int t,n;
	cin>>t;
	while(t--)
	{
		cin>>n;
		for(int i=0;i<n;i++)
		{
			cin>>x>>y;
			v.pb(mp(x,y));
		}
		sort(v.begin() ,v.end());
		min = v[0].second;
		q.push(min);
		for(int i=1;i<n;i++)
		{
            q.push()
		}
	}
	return 0;
}