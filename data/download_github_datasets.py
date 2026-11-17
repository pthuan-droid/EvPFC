"""Download the single-cell multi-omics benchmark data sets hosted on GitHub.
Run:  python data/download_github_datasets.py   (files go to <repo>/datasets or $EVPFC_DATA)
Files already downloaded are skipped (size check)."""
import os, sys, urllib.request, json
ROOT = os.path.abspath(os.environ.get('EVPFC_DATA', os.path.join(os.path.dirname(__file__), '..', 'datasets')))
RAW = 'https://raw.githubusercontent.com/{repo}/{br}/{path}'
FILES = [
 # (target folder, repo, branch, path)
 # --- RNA + ADT (CITE-seq) ---
 ('RNA_ADT/BMNC',            'DayuHuu/scEMC','master','scEMC/datasets/BMNC.mat'),
 ('RNA_ADT/SLN111_D1',       'YosefLab/totalVI_reproducibility','master','data/spleen_lymph_111.h5ad'),
 ('RNA_ADT/SLN206_D2',       'YosefLab/totalVI_reproducibility','master','data/spleen_lymph_206.h5ad'),
 ('RNA_ADT/PBMC10k_CITE',    'YosefLab/totalVI_reproducibility','master','data/pbmc_10k_protein_v3.h5ad'),
 ('RNA_ADT/PBMC5k_CITE',     'YosefLab/totalVI_reproducibility','master','data/pbmc_5k_protein_v3.h5ad'),
 ('RNA_ADT/MALT10k_CITE',    'YosefLab/totalVI_reproducibility','master','data/malt_10k_protein_v3.h5ad'),
 ('RNA_ADT/SLN_annotations', 'YosefLab/totalVI_reproducibility','master','spleen_lymph_all/annotations/annotations_subclustered.csv'),
 # --- RNA + ATAC ---
 ('RNA_ATAC/SMAGE-3K',       'DayuHuu/scUNC','master','data/SMAGE-3K.mat'),
 ('RNA_ATAC/CellMix_SNARE_scCLC','CSUBioGroup/scCLC','main','demo_data/h5ad_dir/SNARE_cellline/data.h5ad'),
 ('RNA_ATAC/CellMix_SNARE_scCLC/ATAC','CSUBioGroup/scCLC','main','demo_data/h5ad_dir/SNARE_cellline_ATAC/data.h5ad'),
 ('RNA_ATAC/CellMix_SNARE_scCLC','CSUBioGroup/scCLC','main','demo_data/SNARE_cellline/SNARE_cellline_cluster.csv'),
 ('RNA_ATAC/CellMix_SNARE_scCLC','CSUBioGroup/scCLC','main','demo_data/SNARE_cellline_cluster_typeName.tsv'),
 ('RNA_ATAC/CellMix_SNARE_DCCA','cmzuo11/DCCA','main','Example_test/scRNA_seq_SNARE.tsv'),
 ('RNA_ATAC/CellMix_SNARE_DCCA','cmzuo11/DCCA','main','Example_test/scATAC_seq_SNARE.txt'),
 ('RNA_ATAC/PBMC3k_labels_DCCA','cmzuo11/DCCA','main','Utilities/pbmc_3k_cell_cluster_2874.txt'),
] + [('RNA_ATAC/Ma-2020-1_features_scMDCL','ubuntu1024/scMDCL','main',f'input/Ma-2020-1/{f}') for f in ['RNA_fea.npy','ATAC_fea.npy','label.npy','RNA_euc_20.npz','ATAC_euc_20.npz']] \
  + [('RNA_ATAC/PBMC-10k_features_scMDCL','ubuntu1024/scMDCL','main',f'input/PBMC-10k/{f}') for f in ['RNA_fea.npy','ATAC_fea.npy','label.npy','RNA_euc_20.npz','ATAC_euc_20.npz']] \
  + [('Mosaic/ASAP-PBMC_scMoMaT','PeterZZQ/scMoMaT','main',f'data/real/ASAP-PBMC/{f}') for f in ['GxC1.npz','GxC2.npz','GxR.npz','PxC1.npz','PxC2.npz','PxC3.npz','PxC4.npz','RxC3.npz','RxC4.npz','genes.txt','regions.txt','meta_c1.csv','meta_c2.csv','meta_c3.csv','meta_c4.csv']] \
  + [('Mosaic/MOp_5batches_scMoMaT','PeterZZQ/scMoMaT','main',f'data/real/MOp_5batches/{f}') for f in ['BxC1.npz','BxC3.npz','BxC5.npz','GxC1.npz','GxC2.npz','GxC4.npz','GxR.npz','MxC3.csv','RxC1.npz','RxC3.npz','RxC5.npz','regions.txt']+[f'meta_c{i}.csv' for i in range(1,6)]] \
  + [('scRNA_only/Quake_10x_Spleen','HKzda210/scMTO','master','data/Quake_10x_Spleen/data.h5')]

def main():
    log=[]
    for d,repo,br,path in FILES:
        out=os.path.join(ROOT,d,os.path.basename(path)); os.makedirs(os.path.dirname(out),exist_ok=True)
        url=RAW.format(repo=repo,br=br,path=path)
        if os.path.exists(out) and os.path.getsize(out)>1000:
            log.append((out,'skip',os.path.getsize(out))); continue
        try:
            urllib.request.urlretrieve(url,out); sz=os.path.getsize(out)
            head=open(out,'rb').read(60)
            st='LFS-POINTER' if head.startswith(b'version https://git-lfs') else 'ok'
            log.append((out,st,sz)); print(st,f'{sz/1e6:.1f}MB',d,os.path.basename(path),flush=True)
        except Exception as e:
            log.append((out,'FAIL '+str(e),0)); print('FAIL',url,e,flush=True)
    json.dump([dict(file=os.path.relpath(a,ROOT),status=b,bytes=c) for a,b,c in log],open(os.path.join(ROOT,'download_log.json'),'w'),indent=1)
if __name__=='__main__': main()
