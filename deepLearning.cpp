#include <iostream>
#include <vector>

using namespace std;

double fout(double x)
{
    return 1 / (1 + exp(-x));
}

class Neuron {
    double u;
    double x;
    vector<double> w;
    double bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0)
    {
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    const vector<double>& getW() const { return w; }
    double getBias() const { return bias; }

    void scalarProduct(const vector<double>& in) {
        if(w.size() == in.size()) {
            u = 0.0;
            for(int i = 0; i < w.size(); i++) {
                u += w[i] * in[i];
            }
            x = fout(u);
        }
    }
};

int main()
{
    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    neurons.resize(5);
    neurons[0].resize(1);
    neurons[1].resize(5);
    neurons[2].resize(3);
    neurons[3].resize(5);
    neurons[4].resize(1);

    vector<double> w(1);
    w[0] = 1.0;
    neurons[0][0] = new Neuron(w, 0.0);

    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i - 1].size(); j++) {
            w[j] = 1.0;
        }
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j] = new Neuron(w, 0.0);
        }
    }

    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
        }
    }


    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}