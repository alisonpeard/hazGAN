
# compare contents of local generated/gaussian dirs
rclone check \
"/Users/alison/Documents/dphil/data/hazGAN/generated/rp10000/gaussian/npy" \
"/Users/alison/Documents/dphil/data/hazGAN/generated/rp10000/gaussianv1/npy"
# all the same 03-02-2026

# compare contents of remote/local generated/gaussian dirs
rclone check \
--sftp-ask-password \
/Users/alison/Documents/dphil/data/hazGAN/generated/rp10000/gaussian/npy \
:sftp,host=gateway.arc.ox.ac.uk,user=spet5107:/data/ouce-opsis/spet5107/data/generated/rp10000/gaussian/npy