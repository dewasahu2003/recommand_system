from matplotlib import pyplot
import pandas

movies=pandas.read_csv('./ml-latest-small/movies.csv')
ratings=pandas.read_csv('./ml-latest-small/ratings.csv')

_data=pandas.merge(movies,ratings,on='movieId')

data=pandas.DataFrame(_data.groupby('title')['rating'].mean())

data['n_rating']=pandas.Series(_data.groupby('title')['rating'].count())

print(data.head())

pyplot.figure(figsize=(10,4))
data['rating'].hist(bins=70)


moviemat=_data.pivot_table(index='userId',columns='title',values='rating')

print('1')
BvS_user_rating=moviemat['Imaginarium of Doctor Parnassus, The (2009)']


similar_to_BvS=moviemat.corrwith(BvS_user_rating)
print(1)
print(similar_to_BvS)
print(2)

corr_BvS=pandas.DataFrame(similar_to_BvS,columns=['Correlation'])
corr_BvS.dropna(inplace=True)
corr_BvS=corr_BvS.sort_values('Correlation',ascending=False)
corr_BvS=corr_BvS.join(data['n_rating'])
print(corr_BvS)