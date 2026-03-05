from app import search

q = 'What is a brain tumor?'
print('Searching via app.search:')
res = search(q, top_k=3)
for i, r in enumerate(res):
    print('---', i)
    print(r[:400])
