#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    2023/12/11: Yuta Watabe
    このモジュールはEKモデルをクラスとして実装するものです。
    This module builds EK model as a class. Basic functions. 
"""

import numpy as np
import pandas as pd
import copy

class Onesector():

    def __init__(self,
                
        year, # 年度 Year of the data
        countries, # 国のリスト List of ISO3 codes
        X, # 貿易フローの行列 Trade flows (N * N ndarray)
        theta = 4, # 貿易弾力性 Trade elasticity  
        ):
 
        self.year,self.countries = year,countries
        self.N = len(countries)
        self.theta,self.X = theta, X

        # 国の生産、消費、貿易赤字を貿易額のフローから計算する。
        # Calculate absorption, production, trade deficit
        self.Xm = np.sum(X,axis=(0))
        self.Ym = np.sum(X,axis=(1))
        self.D  = self.Xm - self.Ym 

        return

    @staticmethod
    def from_parameters(year,countries,theta,T,tau,L,D,):    
        
        """
        パラメータからEKモデルの均衡状態を計算し、この均衡を表す`Onesector`オブジェクトを返します。

        パラメーター:
        - year (int): データの年度。
        - countries (list): ISO3国コードのリスト。
        - theta (float): 貿易弾力性パラメータ。
        - T (ndarray): 技術パラメータ。国の数と同じ長さの配列。
        - tau (ndarray): 氷山貿易コストパラメータ。国と国の間の貿易コストを表す2D配列（N x N）。
        - L (ndarray): 労働供給。国の数と同じ長さの配列。
        - D (ndarray): 貿易赤字。国の数と同じ長さの配列。

        戻り値:
        - Onesector: 計算された均衡状態を表すOnesectorクラスのインスタンス。

        このメソッドは、入力パラメータを使用して均衡の賃金、価格、貿易フローを計算します。

        Calculates the equilibrium state of the EK model given a set of static parameters and returns an `Onesector` object representing this equilibrium.

        Parameters:
        - year (int): The year of the data.
        - countries (list): List of ISO3 country codes.
        - theta (float): Trade elasticity parameter.
        - T (ndarray): Technology parameter, an array with length equal to the number of countries.
        - tau (ndarray): Iceberg trade cost parameter, a 2D array (N x N) representing trade costs between countries.
        - L (ndarray): Labor endowment, an array with length equal to the number of countries.
        - D (ndarray): Trade deficit, an array with length equal to the number of countries.

        Returns:
        - Onesector: An instance of the Onesector class, representing the calculated equilibrium state.

        This method computes the equilibrium wages, prices, and trade flows using the input parameters.
        """

        # 収束パラメーターを設定する
        # Set some convergence parameter
        psi = 0.1 # 収束スピード Convergence speed
        tol = 0.00000001 # 収束閾値 Convergence tolerance
        dif = 1 # 初期の収束判定値 initial convergence criterion 

        # 国の数と、初期の賃金を設定する。世界のGDPが1になるようにしておく。
        # Calculate number of countries and set initial wages. Normalize wages so that world GDP is one.
        N = len(countries)
        w = np.ones(N)
        wgdp = np.sum(w * L)
        w = w / wgdp

        while dif > tol: 

            # 価格を計算する
            # Calculate price
            p = np.zeros((N,N))
            for OR,DE in np.ndindex((N,N)):
                p[OR,DE] = w[OR]* tau[OR,DE]
            
            # pi (輸入シェア) を計算する
            # Calculate pi (trade share)
            pi_num = np.zeros((N,N))
            pi_den = np.zeros((N))
            pi = np.zeros((N,N))
            for OR,DE in np.ndindex((N,N)):
                pi_num[OR,DE] = T[OR] * p[OR,DE]**(-theta)
                pi_den[DE] += pi_num[OR,DE]
            for OR,DE in np.ndindex((N,N)):
                pi[OR,DE] = pi_num[OR,DE] / pi_den[DE]

            P = pi_den**(-1/theta) # 価格指数 Price index
            
            # 超過労働需要の計算 Calculate excess labor demand
            wLS = w * L
            Xm  = wLS + D
            wLD = np.zeros((N))
            for OR,DE in np.ndindex((N,N)):
                wLD[OR] += pi[OR,DE] * Xm[DE]
            ZL = (wLD - wLS) / w
            w = w * (1 + psi / L * ZL)

            # 世界全体のGDPが1になるように賃金を基準化する
            # Normalize wages so that the world GDP is one
            wgdp = np.sum(w * L )
            w = w / wgdp
            dif = np.max(np.abs(ZL))

        # 収束後に貿易フローを計算する Calculate trade flows after convergence
        Xm = w * L + D
        X = np.zeros((N,N))
        for OR,DE in np.ndindex((N,N)):
             X[OR,DE] = pi[OR,DE] * Xm[DE]

        return Onesector(year=year,countries=countries,X=X,theta=theta)
    
    def exacthatalgebra(self,tauhat,to_df=False,scenario=""):
        """
        初期均衡から貿易コストの変更後の新しい均衡へのExact Hat Algebraを行います。

        パラメーター:
        - tauhat (ndarray): 貿易コスト（氷山貿易コスト）の比率変化を表す配列。
        - df (bool): 結果をデータフレームに保存するか
        - scenario (str): 反実仮想のシナリオの名前

        戻り値:
        - tuple: 新しいOnesectorオブジェクトと「帽子」の辞書を含むタプル。
        - Onesectorオブジェクトは、新しい均衡状態を表します。
        - 辞書には「what」 (賃金変化)、「Phat」(価格指数変化)、「rexphat」(実質消費変化)が含まれます。
        - to_dfは反実仮想と変化率をすべて記録したデータフレームになります。

        このメソッドは、貿易コストを更新し、均衡を再計算し、様々な経済変数(賃金、価格など)の比率変化を計算します。
        経済の構造（例えば、技術、労働供給）は貿易コスト以外変わらないと仮定します。

        Performs an exact hat algebra calculation from a steady state to a new steady state after a change in trade costs.

        Parameters:
        - tauhat (ndarray): An array representing the proportional changes in trade costs (iceberg trade costs).
        - to_df (bool): Whether to save the result to df
        - scenario (str): Name of the counterfactual scenario

        Returns:
        - tuple: A tuple containing a new Onesector object and a dictionary of 'hats'.
        - The Onesector object represents the new equilibrium state.
        - The dictionary contains 'what' (wage changes), 'Phat' (price index changes), and 'rexphat' (real expenditure changes).

        This method updates the trade cost, recalculates the equilibrium, and computes the proportional changes in various economic variables (wages, prices, etc.). 
        It assumes that the structure of the economy (e.g., technology, labor endowment) remains unchanged except for the trade costs.
        """

        # 収束パラメーターを設定 
        # Set convergence parameter
        psi = 0.1
        tol = 0.00000001
        
        N = self.N
        
        # pi (輸入シェア） を計算。 
        # Calculate pi (import share)
        pi = np.zeros((N,N))
        for PR,DE in np.ndindex((N,N)):
           pi[PR,DE] = self.X[PR,DE] / self.Xm[DE]
        
        # what (賃金の変化率）の初期値を設定。世界GDPが変化しないように基準化する。
        # Put an initial guess on what (changes in wages)。Keep the world GDP constant.
        what = np.ones(N) 
        wgdp = np.sum(self.Ym)
        wgdp1 = np.sum(what * self.Ym) 
        what = what / wgdp1 * wgdp

        dif = 1
        while dif > tol:

            # 価格、および価格指数の変化の計算
            # Calculate price changes
            phat = np.zeros((N,N))
            Phat = np.zeros((N))
            for OR,DE in np.ndindex((N,N)):
                phat[OR,DE] = what[OR] * tauhat[OR,DE]
                Phat[DE] += pi[OR,DE] * phat[OR,DE]**(-self.theta)
            Phat = Phat ** (-1/self.theta)  
            
            # 輸入シェアの変化を計算
            # Calculate pihat
            pihat = np.zeros((N,N))
            for OR,DE in np.ndindex((N,N)):
                pihat[OR,DE] = phat[OR,DE]**(-self.theta) / Phat[DE]**(-self.theta)

            # 消費と生産の値を更新する
            # Update absorption and supply (value)
            wLS1 = what * self.Ym
            Xm1 = wLS1 + self.D

            # 労働需要を計算する
            # Calculate factor demand
            wLD1 = np.zeros((N))
            for OR,DE in np.ndindex((N,N)):
                wLD1[OR] +=  pi[OR,DE] * pihat[OR,DE] * Xm1[DE]

            # 超過労働需要を計算して、賃金変化率を更新する
            # Calculate excess labor demand and update what
            ZL = (wLD1 - wLS1) / what
            what = what * (1 + psi * ZL / wLS1)

            # GDPがexact hat algebraの前後で変化しないように賃金変化を基準化する
            # Normalize changes in wages so that world GDP is constant.
            wgdp1 = np.sum(what * self.Ym)
            what = what / wgdp1 * wgdp

            dif = np.max(np.abs(ZL))

        # 新しい貿易フローを計算 
        # Calculate new trade flow.
        X1  = np.zeros((N,N))
        for OR,DE in np.ndindex((N,N)):
            X1[OR,DE] = pi[OR,DE] * pihat[OR,DE] * Xm1[DE]
        
        newmodel = Onesector(year=self.year,countries=self.countries,X=X1,
                            theta=self.theta)
            
        rwhat = what / Phat
        rexphat = Xm1 / self.Xm / Phat
        hats = {
            "rwhat":rwhat,
            "rexphat":rexphat,
                }

        if to_df:

            # データフレームとして出力する均衡アウトカムを整理する。
            # Organize equilibrium outomes to dataframe.
            df_X = pd.DataFrame(self.X,index=self.countries,columns=self.countries) \
                     .reset_index().rename(columns={"index":"OR"}) \
                     .melt(id_vars="OR",var_name="DE",value_name="value").assign(variable="export_sq")
            df_X1 = pd.DataFrame(newmodel.X,index=self.countries,columns=self.countries) \
                     .reset_index().rename(columns={"index":"OR"}) \
                     .melt(id_vars="OR",var_name="DE",value_name="value").assign(variable="export_cf")
            df_Xhat = pd.DataFrame(newmodel.X/self.X,index=self.countries,columns=self.countries) \
                     .reset_index().rename(columns={"index":"OR"}) \
                     .melt(id_vars="OR",var_name="DE",value_name="value").assign(variable="export_hat")
            df_rexphat = pd.DataFrame(rexphat,index=self.countries,columns=["value"]).reset_index() \
                        .rename(columns={"index":"OR"}).assign(variable="realexpenditure_hat")
            df_rwhat = pd.DataFrame(rwhat,index=self.countries,columns=["value"]).reset_index() \
                        .rename(columns={"index":"OR"}).assign(variable="realwage_hat")
            
            df = pd.concat([df_X,df_X1,df_Xhat,df_rexphat,df_rwhat],axis=0).assign(scenario=scenario)

            return newmodel,hats,df

        else:
            return newmodel,hats



def test():
    """
    このプログラムはExact Hat Algebraのコードが正しいかテストするものです。
    具体的には、以下の六つのステップでテストします。
     1. ランダムなパラメータからbenchmarkモデルを生成(Onesector.from_static_parameters)する。
     2. ランダムにtauhatを生成し、そこから新しいtau1を計算する。
     3. 新しいtau1の下でneweqmモデルを生成(Onesector.from_static_parameters)する。
     4. Benchmarkモデルからexacthatlagebraを行い(benchmark.exacthatlagebra(tauhat))、新しいneweqmhatモデルを出力する
     5. neweqmhatモデルからtauhatを巻き戻す(1/tauhatを使う)ことで元の均衡(benchmark)に戻るかどうかを確認する。
     6. Exacthatalgebraから出たモデルと、neweqmモデルが一致するかを確認する。

    This program tests the correctness of the Exact Hat Algebra code. Specifically, it tests using the following six steps:
     1. Generate a benchmark model from random parameters (using Onesector.from_static_parameters).
     2. Randomly generate tauhat and then calculate a new tau1 from it.
     3. Generate a neweqm model under the new tau1 (using Onesector.from_static_parameters).
     4. Perform exacthatlagebra from the benchmark model (using benchmark.exacthatlagebra(tauhat)) and output a new neweqmhat model.
     5. Confirm whether unwinding tauhat (using 1 / tauhat) from the neweqmhat model returns to the original equilibrium (benchmark).
     6. Check if the model obtained from Exacthatalgebra and the neweqm model are consistent.
    """

    # 1. ランダムなパラメーターからBenchmark equilibrumを生成する。
    # Simulate benchmark equilibrium from random parameters.
    countries =  ["DEU","JPN","USA"] # 国のリスト List of ISO3 codes: list(N)
    N = len(countries) # 国の数 Number of countries
    year = 3000 # 年度 Year of the data we use: int 
    theta = 4  # 貿易弾力性 Trade elasticity: ndarray(S)
    T0 = np.random.rand((N)) # 技術パラメーター Technology parameter: ndarray(N) T
    tau0 = np.random.rand(N,N) / 10 + 1 # 氷塊貿易費用 Trade cost parameter: ndarray(N,N)
    L0 = np.random.rand((N)) # 労働供給 Labor endowment: ndarray(N)

    # 貿易赤字は世界全体で足して0になるようにしないといけない。
    # Trade deficit must sum up to 0
    D0 = np.random.rand((N)) / 10 
    D0 = D0 - np.mean(D0) # これをすることによって平均0になる。 by doing this, we get trade deficit to sum up to zero.

    benchmark = Onesector.from_parameters(
        countries = countries, year = year, theta = theta,
        T = T0,L = L0,D = D0,
        tau = tau0
        )
    print(benchmark)

    # 2. tauhatをランダムに生成する
    # Randomly generate tauhat
    tauhat = np.random.rand(N,N) / 10 + 1

    # 3. tau0とtauhatから新しいtau1を生成し、新しくモデルを生成する
    # Generate tau1 from tau0 and tauhat, and recalculate the model.
    tau1 = tau0 * tauhat

    neweqm = Onesector.from_parameters(
        countries = countries, year = year, theta = theta,
        T = T0, L = L0, D = D0,
        tau = tau1,
    )

    # 4. Benchmarkモデルからexacthatalgebraを行い(benchmark.exacthatlagebra(tauhat))、neweqmhatモデルを出力する。
    # Perform exact hat algebra from the benchmark model and output newewqmhat.
    neweqmhat,_ = benchmark.exacthatalgebra(tauhat,scenario="test")

    # 4.5 exacthatalgebraを使い、1/tauhatを使うことにより均衡を巻き戻して、元の均衡に一致するかを確認する
    # Reverse the exact hat algebra using 1/tauhat and chekc if the equilibrium goes back to the benchmark equilibrium.
    benchmarkhat,_ = neweqmhat.exacthatalgebra(1/tauhat)
    print(benchmarkhat.X/benchmark.X)

    # 5. Exacthatalgebraから出たモデルと、neweqmモデルが一致するかを確認する。
    # Check whether the result of exact hat algebra and neweqm model (which is resolved in level) coincides.
    print(neweqm.X / neweqmhat.X)

    return

if __name__ == "__main__":
    test()


