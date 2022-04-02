% Script pour optimiser le critere par la methode de descente du gradient a
% pas variable


f0=load('Data/DigitTrain_0.mat');
f1=load('Data/DigitTrain_1.mat');
f2=load('Data/DigitTrain_2.mat');
f3=load('Data/DigitTrain_3.mat');
f4=load('Data/DigitTrain_4.mat');
f5=load('Data/DigitTrain_5.mat');
f6=load('Data/DigitTrain_6.mat');
f7=load('Data/DigitTrain_7.mat');
f8=load('Data/DigitTrain_8.mat');
f9=load('Data/DigitTrain_9.mat');

data_0=f0.imgs;
data_1=f1.imgs;
data_2=f2.imgs;
data_3=f3.imgs;
data_4=f4.imgs;
data_5=f5.imgs;
data_6=f6.imgs;
data_7=f7.imgs;
data_8=f8.imgs;
data_9=f9.imgs;

data=cat(3,data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9); % On concatène l'ensemble des données de toutes les images
data=reshape(data,(size(data,1)*(size(data,2))),size(data,3)); % Nous remettons en forme la matrice data
biais=ones(1,size(data,2));
data=cat(1,data,biais);

C0=zeros(10,size(data_0,3));C0(1,:)=1; % Nous ajoutons une dimension pour spécifier de quel perceptron il s'agit
C1=zeros(10,size(data_1,3));C1(2,:)=1;
C2=zeros(10,size(data_2,3));C2(3,:)=1;
C3=zeros(10,size(data_3,3));C3(4,:)=1;
C4=zeros(10,size(data_4,3));C4(5,:)=1;
C5=zeros(10,size(data_5,3));C5(6,:)=1;
C6=zeros(10,size(data_6,3));C6(7,:)=1;
C7=zeros(10,size(data_7,3));C7(8,:)=1;
C8=zeros(10,size(data_8,3));C8(9,:)=1;
C9=zeros(10,size(data_9,3));C9(10,:)=1;

C=cat(2,C0,C1,C2,C3,C4,C5,C6,C7,C8,C9); % On concatène l'ensemble pour donner une classification "1 contre tous"

% Parametres
rho=10e-1;          
wp=rand(10,size(data,1))-0.5;          
nbItMax =1000;
W=zeros(10,nbItMax,size(data,1));
W(:,1,:)=wp;
z=zeros(10,nbItMax,size(data,2));
y=zeros(10,nbItMax,size(data,2));
gradJ=zeros(10,nbItMax,size(data,1));
for ind=2:nbItMax
    
    for i=1:10
        W1=reshape(W(i,ind-1,:),1,401);
        z(i,ind-1,:)=W1*data;
        y(i,ind-1,:)=1./(1+exp(-z(i,ind-1,:)));
        y1=reshape(y(i,ind-1,:),1,60000);
        gradJ(i,ind-1,:) = (1/size(data,2)).*((((y1-C(i,:)).*(y1-y1.*y1)))*data');
        W(i,ind,:)=W(i,ind-1,:)-(rho*gradJ(i,ind-1,:));
    end

end


f0test=load('Data/DigitTest_0.mat');
f1test=load('Data/DigitTest_1.mat');
f2test=load('Data/DigitTest_2.mat');
f3test=load('Data/DigitTest_3.mat');
f4test=load('Data/DigitTest_4.mat');
f5test=load('Data/DigitTest_5.mat');
f6test=load('Data/DigitTest_6.mat');
f7test=load('Data/DigitTest_7.mat');
f8test=load('Data/DigitTest_8.mat');
f9test=load('Data/DigitTest_9.mat');

data_0test=f0test.imgs;
data_1test=f1test.imgs;
data_2test=f2test.imgs;
data_3test=f3test.imgs;
data_4test=f4test.imgs;
data_5test=f5test.imgs;
data_6test=f6test.imgs;
data_7test=f7test.imgs;
data_8test=f8test.imgs;
data_9test=f9test.imgs;

datatest=cat(3,data_0test,data_1test,data_2test,data_3test,data_4test,data_5test,data_6test,data_7test,data_8test,data_9test);
datatest=reshape(datatest,(size(datatest,1)*(size(datatest,2))),size(datatest,3));
biaistest=ones(1,size(datatest,2));
datatest=cat(1,datatest,biaistest);

C0t=zeros(10,size(data_0test,3));C0t(1,:)=1;
C1t=zeros(10,size(data_1test,3));C1t(2,:)=1;
C2t=zeros(10,size(data_2test,3));C2t(3,:)=1;
C3t=zeros(10,size(data_3test,3));C3t(4,:)=1;
C4t=zeros(10,size(data_4test,3));C4t(5,:)=1;
C5t=zeros(10,size(data_5test,3));C5t(6,:)=1;
C6t=zeros(10,size(data_6test,3));C6t(7,:)=1;
C7t=zeros(10,size(data_7test,3));C7t(8,:)=1;
C8t=zeros(10,size(data_8test,3));C8t(9,:)=1;
C9t=zeros(10,size(data_9test,3));C9t(10,:)=1;

Ct=cat(2,C0t,C1t,C2t,C3t,C4t,C5t,C6t,C7t,C8t,C9t);

W2=reshape(W(:,nbItMax,:),10,401);
z2=W2*datatest;
y2=1./(1.+exp(-z2));

ERR=round(y2);
s=0;
p=0;


% Ici nous affichons la matrice de confusion pour chaque perceptron.
figure(2);
subplot(2,5,1);
l0=confusionmat(Ct(1,:),ERR(1,:));
confusionchart(l0);
subplot(2,5,2);
l1=confusionmat(Ct(2,:),ERR(2,:));
confusionchart(l1);
subplot(2,5,3);
l2=confusionmat(Ct(3,:),ERR(3,:));
confusionchart(l2);
subplot(2,5,4);
l3=confusionmat(Ct(4,:),ERR(4,:));
confusionchart(l3);
subplot(2,5,5);
l4=confusionmat(Ct(5,:),ERR(5,:));
confusionchart(l4);
subplot(2,5,6);
l5=confusionmat(Ct(6,:),ERR(6,:));
confusionchart(l5);
subplot(2,5,7);
l6=confusionmat(Ct(7,:),ERR(7,:));
confusionchart(l6);
subplot(2,5,8);
l7=confusionmat(Ct(8,:),ERR(8,:));
confusionchart(l7);
subplot(2,5,9);
l8=confusionmat(Ct(9,:),ERR(9,:));
confusionchart(l8);
subplot(2,5,10);
l9=confusionmat(Ct(10,:),ERR(10,:));
confusionchart(l9);

for ind=1:size(datatest,2)
     if ERR(10,ind)==Ct(10,ind);
         s=s+1;
     else
         p=p+1;
     end

end
TAU=p/size(datatest,2)