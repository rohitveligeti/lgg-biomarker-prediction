from dataset import *
import pandas
import numpy

df = aggregrate()

rows2 = list(df.index.values)
nr2 = []

for r in rows2:
    nr2.append(r + "-01")

df.index = nr2

# print(df)

imm = pandas.read_csv('immune.txt', delimiter='\t', index_col=0)
imm = imm[imm['CancerType'] == 'LGG']
imm = imm.drop(['CancerType', 'P.value', 'Correlation', 'RMSE'], axis=1)
imm = imm.dropna(axis=1)

rows = list(imm.index.values)

new_imm_rows = []

for i in rows:
    i2 = i.split('.')
    new_imm_rows.append(i2[0] + '-' + i2[1] + '-' + i2[2] + "-01")

imm.index = new_imm_rows

imm = imm.reset_index()
imm = imm.drop_duplicates(subset='index')
imm = imm.set_index('index')


# print(new_imm_rows)

# print(imm)

mb = pandas.read_csv('microbiome.txt', delimiter='\t', index_col=0)
mb = mb.drop(['NAME', 'DESCRIPTION', 'URL'], axis =1).T
mb = mb.dropna(axis=1)

# print(mb)

mrna = pandas.read_csv('mrna.txt', delimiter='\t', index_col=1)
mrna = mrna.drop(['Hugo_Symbol'], axis=1).T
mrna = mrna.dropna(axis=1)


# print(mrna)

rppa = pandas.read_csv('rppa.txt', delimiter='\t', index_col=0).T
rppa = rppa.drop(['CASP3|Caspase-3', 'CASP9|Caspase-9', 'PARP1|PARP1'], axis=1)
rppa = rppa.dropna(axis=1)

# print(rppa)

#Part 2 -------------------

df_set = set(df.index.values.tolist())
imm_set = set(imm.index.values.tolist())
mb_set = set(mb.index.values.tolist())
mrna_set = set(mrna.index.values.tolist())
rppa_set = set(rppa.index.values.tolist())

mega_intersect = list(df_set.intersection(imm_set).intersection(mb_set).intersection(mrna_set).intersection(rppa_set))



df = df.loc[mega_intersect]
imm = imm.loc[mega_intersect]
mb = mb.loc[mega_intersect]
mrna = mrna.loc[mega_intersect]
rppa = rppa.loc[mega_intersect]



ndf = pandas.concat([df, imm, mb, mrna, rppa], axis=1)

y = ndf["Outcome"]
ndf = ndf.drop(['Outcome'], axis=1)

print(ndf)
ndf = ndf.loc[:,~ndf.columns.duplicated()]



def corr(x, y):
    """Weighted Covariance"""
    return abs(numpy.sum(((x - numpy.mean(x)) / numpy.std(x)) * ((y - numpy.mean(y)) / numpy.std(y))) / (len(x) - 1))

l1 = list(ndf.columns)
l2 = l1.copy()

print(len(l1), len(set(l1)))

i = 0
io = 0

for one in l1:
    for two in l2:

        bound = 2

        p1 = ndf[one].to_numpy()
        p2 = ndf[two].to_numpy()

        # print(one, two)

        if type(one) == int or type(two) == int:
            bound = 0.2





        amount = corr(p1, p2)**bound

        if amount > 0.7 and amount < 1:
            if one == two:
                continue
            else:
                io += 1
                # i, io, amount, one, 
                print(i, io, round(io / i, 6), amount, one, two)

        i += 1
