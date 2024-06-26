import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import com.google.protobuf.gradle.*

// https://github.com/gradle/kotlin-dsl-samples/tree/master/samples

plugins {
    kotlin("jvm") version "1.7.10"
    `maven-publish`
    id("com.google.protobuf") version "0.8.19"
    application
}

group = "co.marieai"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://jitpack.io")
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("io.grpc:grpc-kotlin-stub:1.3.0")
    implementation("io.grpc:grpc-protobuf:1.50.2")
    implementation("com.google.protobuf:protobuf-kotlin:3.21.9")

    // These are needed by gRPC
    implementation("io.grpc:grpc-netty:1.50.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClass.set("MainKt")
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:3.21.5"
    }
    plugins {
        id("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.49.0"
        }
        id("grpckt") {
            artifact = "io.grpc:protoc-gen-grpc-kotlin:1.3.0:jdk8@jar"
        }
    }
    generateProtoTasks {
        all().forEach {
            it.plugins {
                id("grpc")
                id("grpckt")
            }
            it.builtins {
                id("kotlin")
            }
        }
    }
}

sourceSets {
    main {
        java {
            srcDirs(
                "src/main/java",
                "build/generated/source/proto/main/java", "build/generated/source/proto/main/grpc",

                "src/main/kotlin",
                "build/generated/source/proto/main/kotlin", "build/generated/source/proto/main/grpckt"
            )
        }
    }
}

tasks.withType<JavaCompile> {
    sourceCompatibility = "1.8"
    targetCompatibility = "1.8"
}


val sourcesJar by tasks.creating(Jar::class) {
    archiveClassifier.set("sources")
    dependsOn("generateProto")
    from(sourceSets["main"].java.srcDirs)
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            groupId = "co.marieai"
            artifactId = "marie-ai-client"
            version = "1.0-SNAPSHOT"
            artifact(sourcesJar)
            from(components["java"])
        }
    }
}
