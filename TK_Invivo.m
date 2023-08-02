%% 



clc
clear all
close all

%% Load file
load('Brain2D');

%% Parameters
FOV=256;
Nc = 12;
Nx =  FOV;
Ny =  FOV;
rate = 4;
M=rate;
mask=zeros(Nx,Ny);
mask(1:M:end,:)=1;


%% Normalization
min_a = min(min(DATA(:)));
max_a = max(max(DATA(:)));
for n=1:Nc
    norm(:,:,n) = (DATA(:,:,n)-min_a)./abs(max_a-min_a); 
end 

%% Coil images
coil_img=ifftshift(ifft2(ifftshift(norm)));


figure(1),
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(coil_img(:,:,n)),[])
end

%% 32x32 sample

Sample_from_k=zeros(40,40,Nc);
Sample_from_k(:,:,:)=norm(109:148,109:148,:);
 
%% Hamming window filteration
w = hann(40)*hann(40)';
FFT = fftshift(fft2(w)); % complex matrix
FFT_abs = abs(FFT); % absolut values of fft
imagesc(1+log10(FFT_abs)) % show fft results
w_new = ifft2(ifftshift(FFT)); % you need fft not fft_abs
figure,
imshow(abs(w_new),[])


% Dot multiplication of window with sample

for n=1:Nc
filtered_sample(:,:,n)=w_new.*Sample_from_k(:,:,n);
end

% fitting 32x32 in back and zero padding
padded_filter=zeros(Nx,Ny,Nc);
padded_filter(109:148,109:148,:)=filtered_sample(:,:,:);
 

%% Coil images for coil sensitivity from sample (filtered)
for n=1:Nc
    coil_img1(:,:,n)=ifftshift(ifft2(ifftshift(padded_filter(:,:,n))));   
end
figure(1),
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(coil_img1(:,:,n)),[])
end

%% Combined SOS image (low resolution)for coil sensitivty
for n=1:Nc
squared_img(:,:,n) = power(abs(coil_img1(:,:,n)), 2);
end

sum_img = sum(squared_img, 3);
rsos1 = sqrt(sum_img);
figure,
imshow((abs(rsos1)),[])

%% Reference Image for error calculation and difference Image
for n=1:Nc
sq_img(:,:,n) = power(abs(coil_img(:,:,n)), 2);
end
s_img = sum(sq_img, 3);
xo = sqrt(s_img);
figure,
imshow((abs(xo)),[])

%% Coil sensitivity

for n=1:Nc
c_sens(:,:,n)=coil_img1(:,:,n)./rsos1;
end
figure,
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(c_sens(:,:,n)),[])
end


%% Undersampling normalized data
norm = mask.*norm;

%% ifft noisy k space (image domain)

noise_along_y = ifftshift(ifft2(ifftshift(norm*M))); 
figure
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(noise_along_y(:,:,n)),[])
end

%% Noise covariance matrix from noise sample (256x10x12)
noise_sample=noise_along_y(1:256,1:10,:);
permute_noise_sample=permute(noise_sample,[1 2 3]);
reshape_noise_sample=reshape(permute_noise_sample,[],Nc);
psi=(reshape_noise_sample'*reshape_noise_sample);
figure,
imagesc(real(psi));
colorbar

%% y tilda (Equation 3 of paper)
[V,D,W] = eig(psi);
first=D^(-1/2)*W';
permute_noise_along_y=permute(noise_along_y,[1 2 3]);
reshape_noise_along_y=reshape(permute_noise_along_y,[],Nc);
ytil=first*reshape_noise_along_y';

permute_ytil=permute(ytil,[3 2 1]);
reshape_ytil=reshape(permute_ytil,256,256,Nc);

figure,
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(reshape_ytil(:,:,n)),[])
end

%% A tilda (Equation 3 of paper)
permute_sens=permute(c_sens,[1 2 3]);
reshape_sens=reshape(permute_sens,[],Nc);

Atil=first*reshape_sens';

permute_atil=permute(Atil,[3 2 1]);
reshape_atil=reshape(permute_atil,256,256,Nc);

figure
for n=1:Nc
    subplot(2,ceil(Nc/2),n)
    imshow(abs(reshape_atil(:,:,n)),[])
end

%% Normalizing A tilda again because lamda is high overwise
minimum = min(min(reshape_atil(:)));
maximum = max(max(reshape_atil(:)));
for n=1:Nc
    norm_atil(:,:,n) = (reshape_atil(:,:,n)-minimum)./abs(maximum-minimum); 
end 


%% Pre-whitened image without regularization
delta=Ny/M;
test=zeros(Nx,Ny);

for x=1:Nx
     for y=1:delta
          for L=1:Nc
              B(L,1:M)=reshape_atil(y:delta:end,x,L);
              pixel_vector(L,1)=reshape_ytil(y,x,L);
          end
          invB=pinv(B);
          test(y:delta:end,x)=invB*pixel_vector;
     end
end

figure,
imshow(abs(test),[])

% Difference Image
diff=abs(xo)-abs(test);   %%if A tilda is again normalized it will not show correct diff image because xo was constructed from one time normalization, mismatch between data 
figure,
imshow(abs(diff),[])

% Error  
error = (abs(xo)-abs(test)).^2;   %%if A tilda is again normalized it will not show correct error because xo was constructed from one time normalization, mismatch between data 
RMSE = sqrt(sum(error(:))/(Nx * Ny));
NRMSE = RMSE/(Nx*Ny)


%% without pre-whitening

delta=Ny/M;
recon_imge=zeros(Nx,Ny);

for x=1:Nx
     for y=1:delta
          for L=1:Nc
              B(L,1:M)=c_sens(y:delta:end,x,L);
              pixel_vector(L,1)=noise_along_y(y,x,L);
          end
          invB=pinv(B);
          recon_imge(y:delta:end,x)=invB*pixel_vector;
     end
end

figure,
imshow(abs(recon_imge),[])

% Error
error = (abs(xo)-abs(recon_imge)).^2;
RMSE = sqrt(sum(error(:))/(Nx * Ny));
NRMSE = RMSE/(Nx*Ny)

% Difference Image 
diff=abs(xo)-abs(recon_imge);
figure,
imshow(abs(diff),[])


%% Tikhonov Regularization
para=0^2;              %% small lamda values will work
delta=Ny/M;
recon_imge=zeros(Nx,Ny);

for x=1:Nx
     for y=1:delta
          for L=1:Nc
              A(L,1:M)=norm_atil(y:delta:end,x,L);
              pixel_vector(L,1)=reshape_ytil(y,x,L);
          end
          A1=A'*A;
          A2=para*eye(size(A1));
          A3=pinv(A1+A2);
          A4=(A'*pixel_vector)+(para*rsos1(y:delta:end,x));
          recon_imge(y:delta:end,x)=A3*A4;
     end
end

figure,
imshow(abs(recon_imge),[])

% Error
error = (abs(xo)-abs(recon_imge)).^2; %%if A tilda is again normalized it will not show correct error because xo was constructed from one time normalization, mismatch between data
RMSE = sqrt(sum(error(:))/(Nx * Ny));
NRMSE = RMSE/(Nx*Ny)

% Difference Image
diff=abs(xo)-abs(recon_imge); %%if A tilda is again normalized it will not show correct diff image because xo was constructed from one time normalization, mismatch between data
figure,
imshow(abs(diff),[])
figure,
imagesc(abs(diff))
colorbar

%% L curve
% Prior Error
delta=Ny/M;
summation=0;
summation2=0;
summation3=0;
in=1;

for reg_para=0:0.05:10; %% same for all reduction factor
   for x=1:Nx
     for y=1:delta
          for L=1:Nc
              A(L,1:M)=norm_atil(y:delta:end,x,L); % A tilda normalized again (two times)
              pixel_vector(L,1)=reshape_ytil(y,x,L); % One time normalized at beginning
          end

          [U,S,V]=svd(A);
          ite=y:delta:256;
               for n=1:rank(S)
                    Fj=S(n,n).^2/(((S(n,n)).^2)+reg_para.^2);
                          numerator=U(:,n)'*pixel_vector;
                          denominator=numerator/S(n,n);
                          whole=(Fj*(denominator-xo(ite(:,n))))^2;
                          summation=summation+whole; 
                          whole=[];
               end
                            summation2=summation2+summation;
                            summation=0;
     end
                         summation3=summation3+summation2;
                         summation2=0;
   end

                        prior_error(1,in)=summation3;
                        summation3=0;
                        in=in+1;
end

% Model Error

delta=Ny/M;
summation_model=0;
summation2_model=0;
summation3_model=0;
in=1;

for reg_para=0:0.05:10; % same always
for x=1:Nx
     for y=1:delta
          for L=1:Nc
              A(L,1:M)=norm_atil(y:delta:end,x,L);
              pixel_vector(L,1)=reshape_ytil(y,x,L);
          end
%           A1=A'*A;
%           A2=para*eye(size(A1));
%           A3=pinv(A1+A2);
%           A4=(A'*pixel_vector)+(para*xo(y:delta:end,x));
%           recon_imge(y:delta:end,x)=A3*A4;
[U,S,V]=svd(A);

ite=y:delta:256;
for n=1:rank(S)
    Fj=S(n,n).^2/(((S(n,n)).^2)+reg_para.^2);
          half=U(:,n)'*pixel_vector;
          complete=((1-Fj)*half)^2;
          summation_model=summation_model+complete; 
          complete=[];
end
summation2_model=summation2_model+summation_model;
summation_model=0;
     end
     summation3_model=summation3_model+summation2_model;
     summation2_model=0;
end


model_error(1,in)=summation3_model;
in=in+1;
summation3_model=0;
end

% L-curve
figure,
plot(abs(prior_error),abs(model_error))

%% Gmap
% Normalize k space data again


minim = min(min(reshape_ytil(:)));
maxim = max(max(reshape_ytil(:)));
for n=1:Nc
    norm_ytil(:,:,n) = (reshape_ytil(:,:,n)-minim)./abs(maxim-minim); 
end 

para=0^2;  
delta=Ny/M;
gmap=zeros(Nx,Ny);

for x=1:Nx
     for y=1:delta
          for L=1:Nc
              A(L,1:M)=reshape_atil(y:delta:end,x,L); % one time norm
              pixel_vector(L,1)=norm_ytil(y,x,L);  % two times norm
          end
          A1=A'*A;
          A2=para*eye(size(A1));
          A3=pinv(A1+A2);
          A4=(A'*pixel_vector);
          gmap(y:delta:end,x)=A3*A4;
     end
end

figure,
imshow(abs(gmap),[])

diff=abs(xo)-abs(gmap);
figure,
imshow(abs(diff),[])

figure,
imagesc(abs(diff))
