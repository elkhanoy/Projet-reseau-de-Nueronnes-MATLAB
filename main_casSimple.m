% Script pour optimiser le critere par la methode de descente du gradient a
% pas variable


f=load('DataSimulation/DataTrain_2Classes_Perceptron.mat');
C=f.c;
data=ones(3,1000);
data_0=f.data;
data(1:2,1:1000)=data_0;

% Parametres
rho=1*10e0;            % Valeur fixe du pas
wp=ones(1,3);           % On initialise les poids à une valeur 1
nbItMax =1000;        % Le nombre d'itérations
W=zeros(nbItMax,3);     % On crée la matrice contenant les poids pour chaque itération
W(1,:)=wp;              % On insère les poids initiaux dans la matrice
z=zeros(nbItMax,1000);  % On initialise les matrices z et y pour le calcul
y=zeros(nbItMax,1000);
gradJ=zeros(nbItMax,3); % IDEM

for ind = 2:nbItMax % Itérations pour le calcul du gradient
                    
   z(ind-1,:)=W(ind-1,:)*data;
   y(ind-1,:)=1./(1+exp(-z(ind-1,:)));
   gradJ(ind-1,:) = (1/1000).*(((y(ind-1,:)-C).*((y(ind-1,:)-y(ind-1,:).*y(ind-1,:))))*data')  ;
   W(ind,:)=W(ind-1,:)-(rho*gradJ(ind-1,:));
   
end
figure(1),plot(W(:,2),W(:,1),'xr');


f1=load('DataSimulation/DataTest_2Classes_Perceptron.mat');
CTest=f1.cTest;
dataTest=f1.dataTest;
dataTest(3,:)=ones(1,1000);
W1=W(nbItMax,:);
z1=W1*dataTest;
y1=1./(1.+exp(-z1));
%err=abs(CTest-y1)
ERR=round(y1);
s=0;
p=0;
for ind=1:1000 %Calcul du taux d'erreur
    if ERR(ind)==CTest(ind)
        s=s+1;
    else
        p=p+1;
    end

end
TAU=p/1000

[X,Y]=meshgrid(-13:13,-13:13);
meshc(X,Y,1./(1+exp(-X*(-W1(1))-Y*W(2)-W(3)))); % Affichage de Y avec les poids optimaux dans [-12:12]*[-12:12]
grid on
