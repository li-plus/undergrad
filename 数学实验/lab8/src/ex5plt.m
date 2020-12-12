sx = 80;
sy = 50;
r = 0.4;
mu = [0 0];
sigma = [sx^2  r * sx * sy; r * sx * sy  sy^2];

x = (-200:200)';
y = (-200:200)';
len = length(x);
[X, Y] = meshgrid(x, y);
Z = mvnpdf([reshape(X,len^2,1) reshape(Y,len^2,1)], mu, sigma);
Z = reshape(Z, len, len);
mesh(X, Y, Z);
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
