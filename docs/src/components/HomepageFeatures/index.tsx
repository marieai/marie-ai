import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Universal',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
          <ul>
              <li>Build applications that deliver insights from multiple data types such as text, image, audio</li>
              <li>Support all mainstream deep learning frameworks</li>
              {/*<li>Polyglot gateway that supports gRPC, Websockets, HTTP with TLS.</li>*/}
          </ul>
      </>
    ),
  },
  {
    title: 'Ecosystem',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
          <ul>
              <li>Support all mainstream deep learning frameworks</li>
              <li>Prebuild Executors with state of the art machine learning models</li>
          </ul>
      </>
    ),
  },
  {
    title: 'Cloud-native',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
          <ul>
              <li>Seamless Docker container integration</li>
              <li>Deployment to Kubernetes, Docker Compose</li>
              <li>Observability via Prometheus and Grafana.</li>
          </ul>
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {/*<Svg className={styles.featureSvg} role="img" />*/}
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
