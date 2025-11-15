import pandas as pd

def summarize(path, thr=0.5):
    df = pd.read_csv(path)
    n = len(df)
    succ = (df['confidence'] >= thr).sum()
    mean_moves = df['moves'].mean()
    mean_moves_succ = df.loc[df['confidence'] >= thr, 'moves'].mean() if succ>0 else float('nan')
    mean_pixels = df['pixels_seen'].mean()
    mean_conf = df['confidence'].mean()
    print(path)
    print(' episodes', n)
    print(' successes', succ, f'({succ/n*100:.1f}%)')
    print(' mean confidence', f"{mean_conf:.3f}")
    print(' mean moves overall', f"{mean_moves:.1f}")
    print(' mean moves (successes)', f"{mean_moves_succ:.1f}")
    print(' mean pixels seen', f"{mean_pixels:.1f}")
    print('\n')

if __name__ == '__main__':
    summarize('active_explorer_mnist/flood_results.csv', thr=0.5)
    summarize('active_explorer_mnist/2M_results.csv', thr=0.5)
