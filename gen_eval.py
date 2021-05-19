with open('eval_test.csv', 'w') as f:
    for i in range(0, 10000):
        f.write(f'{i},data/tiny-imagenet-200/test/images/test_{i}.JPEG,64,64,3\n')