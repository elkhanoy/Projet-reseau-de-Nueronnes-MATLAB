% Script pour optimiser le critere par la methode de descente du gradient a
% pas fixe pour 2000 données.


f=load('DataSimulation/DataTrain_2Classes_Perceptron_2.mat');
C=f.c;
data=ones(3,2000); % On crée une matrice nous permettant de stocker les données + le biais
data_0=f.data; % Ici matrice 2*2000
data(1:2,1:2000)=data_0; % Ici matrice 3*2000 avec en 3ème ligne le biais = 1

% Parametres
rho=1*10e0;              % Pas fixe
wp=ones(1,3);            % Initialisation des poids
nbItMax =10000;          % Nb d'itérations pour le calcul des poids optimaux
W=zeros(nbItMax,3);      % Matrice permettant de stocker les poids W à chaque itération
W(1,:)=wp;

for ind = 2:nbItMax % Boucle pour l'optimisation
                    
   z(ind-1,:)=W(ind-1,:)*data; % On calcule la valeur Z pour chaque jeu de données
   y(ind-1,:)=1./(1+exp(-z(ind-1,:))); % On l'insère ensuite dans la fonction d'activation
   gradJ(ind-1,:) = (1/2000).*(((y(ind-1,:)-C).*((y(ind-1,:)-y(ind-1,:).*y(ind-1,:))))*data'); % Calcul du gradient
   W(ind,:)=W(ind-1,:)-(rho*gradJ(ind-1,:)); % Calcul du poids optimisé à la ind-ème itération
   
end

%%%% PARTIE DU CODE PERMETTANT DE FAIRE LE TEST %%%
f1=load('DataSimulation/DataTest_2Classes_Perceptron_2.mat');
CTest=f1.cTest
dataTest=f1.dataTest
dataTest(3,:)=ones(1,2000)
W1=W(nbItMax,:);
z1=W1*dataTest
y1=1./(1.+exp(-z1))
ERR=round(y1) % On utilise la fonction round() pour trancher entre 0 et 1
s=0
p=0
for ind=1:2000
    if ERR(ind)==CTest(ind)
        s=s+1
    else
        p=p+1
    end

end
TAU=p/2000

figure(1);
subplot(1,2,1);
plot(gradJ,'DisplayName','gradJ')
subplot(1,2,2);
plot(W(:,2),W(:,1),'xr');

