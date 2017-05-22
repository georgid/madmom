# this script is not functional, just a suggestion, which Ajay modified later

pattern = obsModel(1); % (1) is duyek. TODO: do same for other 2 usul patterns 
for i = 1:pattern.barGrid

gmm = pattern.model(i);
gmm = gmm{1} % because of cell
sigma = gmm.Sigma;
mu = gmm.mu;
weights = gmm.PComponents;
gmm_struct = struct('mu',{mu},'sigma',{sigma}, 'weights', {weights})

FileName = strcat(pattern.usul, '_gmm_', int2str(i))
savejson('gmm', gmm_struct, 'FileName', FileName)
end