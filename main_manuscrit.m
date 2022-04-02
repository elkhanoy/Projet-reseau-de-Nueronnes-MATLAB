% Script pour optimiser le critere par la methode de descente du gradient a
% pas variable


f1=load('Data/DigitTrain_1.mat');
f2=load('Data/DigitTrain_9.mat');

data_0=f1.imgs;
data_1=f2.imgs;

data=cat(3,data_1,data_0);
%Nous concaténons les deux matrices de données correspondant aux 2
%chiffres, puis nous convertissons la matrice 3D en matrice 2D avec reshape
data=reshape(data,(size(data,1)*(size(data,2))),size(data,3));
biais=ones(1,size(data,2)); % On ajoute la rangée des biais
data=cat(1,data,biais);

C=ones(1,size(data,2)); % On crée la matrice des valeurs attendues
D=zeros(1,size(data_1,3));
C(1:size(data_1,3))=D;


rho=10e-2;              
wp=rand(1,size(data,1))-0.5;     % Initialisation des poids à des valeurs aléatoires entre -0.5 et 0.5       
nbItMax =300;                    
W=zeros(nbItMax,size(data,1));
W(1,:)=wp;
z=zeros(nbItMax,size(data,2));
y=zeros(nbItMax,size(data,2));
gradJ=zeros(nbItMax,size(data,1));
for ind=2:nbItMax

    z(ind-1,:)=W(ind-1,:)*data;
    y(ind-1,:)=1./(1+exp(-z(ind-1,:)));
    gradJ(ind-1,:) = (1/size(data,2)).*(((y(ind-1,:)-C).*((y(ind-1,:)-y(ind-1,:).*y(ind-1,:))))*data');
    W(ind,:)=W(ind-1,:)-(rho*gradJ(ind-1,:));

end

f1test=load('Data/DigitTest_1.mat');
f2test=load('Data/DigitTest_9.mat');

data_0test=f1test.imgs;
data_1test=f2test.imgs;

datatest=cat(3,data_1test,data_0test);
datatest=reshape(datatest,(size(datatest,1)*(size(datatest,2))),size(datatest,3));
biaistest=ones(1,size(datatest,2));
datatest=cat(1,datatest,biaistest);

Ctest=ones(1,size(datatest,2));
Dtest=zeros(1,size(data_1test,3));
Ctest(1:size(data_1test,3))=Dtest;

W1=W(nbItMax,:);
z1=W1*datatest;
y1=1./(1.+exp(-z1));
ERR=round(y1);
s=0;
p=0;
for ind=1:size(datatest,2)
     if ERR(ind)==Ctest(ind);
         s=s+1;
     else
         p=p+1;
     end

end
TAU=p/size(datatest,2)

l=confusionmat(Ctest,ERR);
confusionchart(l); % Affichage de la matrice de confusion.