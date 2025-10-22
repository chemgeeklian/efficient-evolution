import os
from amis_new import reconstruct_multi_models

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Recommend substitutions to a wildtype sequence'
    )
    parser.add_argument('sequence', type=str,
                        help='Wildtype sequence')
    parser.add_argument(
        '--model-names',
        type=str,
        default=[ 'esm2_t6_8M','esm2_t12_35M' ],
        nargs='+',
        help='Type of language model (e.g., esm1b, esm1v1)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='alpha stringency parameter'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='cuda',
        help='cuda device to use'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if ":" in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda.split(':')[-1]

    mutations_models = reconstruct_multi_models(
        args.sequence,
        args.model_names,
        alpha=args.alpha,
    )
    print(mutations_models)

# python bin/recommend.py MQWQTKLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEQSIPAIVDAYGDKALIGAGTVLKPEQVDALARMGCQLIVTPNIHSEVIRRAVGYGMTVCPGCATATEAFTALEAGAQALKIFPSSAFGPQYIKALKAVLPSDIAVFAVGGVTPENLAQWIDAGCAGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVQ

# python src/recommend.py MQWQTKLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEQSIPAIVDAYGDKALIGAGTVLKPEQVDALARMGCQLIVTPNIHSEVIRRAVGYGMTVCPGCATATEAFTALEAGAQALKIFPSSAFGPQYIKALKAVLPSDIAVFAVGGVTPENLAQWIDAGCAGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVQ